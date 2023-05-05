import numpy as np
import torch
import pandas as pd

from utils.dataloader  import FaceDataset, preprocess, postprocess
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, roc_auc_score, roc_curve, recall_score, mean_absolute_error, r2_score


def training(model, n_epoch, train_loader, optimizer, criterion, validation_loader=None, full_train=False, is_classification=False, is_binary=False, patient=0, DEVICE='cpu', best_save=False, do_save=True, file_name=''):
    """
        return:
            - model
            - train history (mean_losses, std_losses)
            - validation history (mean_losses, std_losses)
    """
    model.to(DEVICE)

    best_val_acc = np.inf
    early_stopping = 0
    stop = False

    history_train_losses = []
    history_validation_losses = []

    best_model = model
    model_patient = model
    best_epoch = 0
    best_score = 0

    learning_scores = []

    for i_epoch in range(n_epoch):
        train_losses = []
        val_losses = []

        model.train()
        print(f'Training process epoch {i_epoch+1}/{n_epoch}')
        for batch in tqdm(train_loader):
            images, targets = batch

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            if not is_classification or is_binary:
                targets = targets.float()

            optimizer.zero_grad()

            predicts = model(images)
            
            if not is_classification or is_binary:
                predicts = torch.reshape(predicts, (predicts.size()[0], ))

            loss = criterion(predicts, targets)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())
        
        mean_train = np.mean(train_losses)

        if not full_train or validation_loader is not None:
            model.eval()
            y_pred = []
            y_true = []
            print(f'Test process {i_epoch+1}/{n_epoch}')
            for batch in tqdm(validation_loader):
                images, targets = batch


                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                targets = targets.float()

                with torch.no_grad():
                    predicts = model(images)
                    predicts = torch.reshape(predicts, (predicts.size()[0],))
                
                loss = criterion(predicts, targets)

                predicts_cpu = predicts.cpu().numpy().flatten()
                targets_cpu = targets.cpu().numpy().flatten()

                if is_binary:
                    targets_cpu[targets_cpu >= 0.5] = 1.0
                    targets_cpu[targets_cpu < 0.5] = 0.0
                    predicts_cpu[predicts_cpu >= 0.5] = 1.0
                    predicts_cpu[predicts_cpu < 0.5] = 0.0

                y_pred = np.append(y_pred, predicts_cpu)
                y_true = np.append(y_true, targets_cpu)

                val_losses.append(loss.item())

            mean_validation =  np.mean(val_losses)

            if patient > 0:
                if early_stopping == 0:
                    model_patient = model

                if early_stopping == patient:
                    stop = True

                if mean_validation < best_val_acc:
                    best_val_acc = mean_validation
                    early_stopping = 0
                else:
                    early_stopping += 1

            
            if is_classification:
                valid_score = accuracy_score(y_true, y_pred)
                scores = f'Accuracy:{valid_score}, F1-Score{f1_score(y_true, y_pred)}'
            else:
                valid_score = r2_score(y_true, y_pred)
                scores = f'MAE: {mean_absolute_error(y_true, y_pred)}, R2 {valid_score}'


            print('[-] epoch {:}/{:}, train loss {:.6f}, valiation loss {:.6f}, socres => {}'.format(
            i_epoch+1, n_epoch, mean_train, mean_validation, scores))

            history_validation_losses.append((mean_validation, np.std(val_losses)))
            print(" ")

            learning_scores.append(valid_score)

            if best_save:
                if best_score < valid_score:
                    best_epoch = i_epoch + 1
                    best_model = model
                    best_score = valid_score
                
        print('[-] epoch {:}/{:}, train loss {:.6f}'.format(
        i_epoch+1, n_epoch, mean_train))

        history_train_losses.append((mean_train, np.std(train_losses)))
        print(" ")

        if stop:
            break
    
    if stop:
        model = model_patient

    if do_save:
        torch.save(model, f'models/{file_name}.pt')
        if not full_train and best_save:
            torch.save(best_model, f'models/{file_name}_best_epoch{best_epoch}.pt')

    return model, np.array(history_train_losses), np.array(history_validation_losses), learning_scores




def testing(model, criterion, test_validation, is_binary=False, is_classification=False, DEVICE='cpu'):
    model.eval()
    model.to(DEVICE)

    progress_test_dataloader = tqdm(test_validation, desc=f"Test")

    images_correct = []
    label_correct = []
    images_incorrect = []
    label_incorrect = []

    val_losses = []
    num_save_correct = 0
    num_save_incorrect = 0

    idx_worst_case = []

    y_pred = []
    y_true = []

    for batch in progress_test_dataloader:
        images, targets = batch

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        if not is_classification or is_binary:
            targets = targets.float()

        with torch.no_grad():
            predicts = model(images)
            predicts = torch.reshape(predicts, (predicts.size()[0], ))
        
        loss = criterion(predicts, targets)
        val_losses.append(loss.item())

        predicts_cpu = predicts.cpu().numpy().flatten()
        targets_cpu = targets.cpu().numpy().flatten()

        if is_binary:
            predicts_cpu[predicts_cpu >= 0.5] = 1.0
            predicts_cpu[predicts_cpu < 0.5] = 0.0
            targets_cpu[targets_cpu >= 0.5] = 1.0
            targets_cpu[targets_cpu < 0.5] = 0.0
        
        if is_classification:
            faceDataset = FaceDataset(preprocess(), is_classification=True)
            label_names = faceDataset.get_all_age_range()
            predicts_cpu = np.int32(predicts_cpu)

        y_pred = np.append(y_pred, predicts_cpu)
        y_true = np.append(y_true, targets_cpu)


        rand_idx = torch.randint(0, len(targets), (1,))

        if predicts_cpu[rand_idx[0]] == targets_cpu[rand_idx[0]]:
            images_correct.append(images[rand_idx[0]].cpu())
            label_n = np.int64(predicts_cpu[rand_idx[0]])
            if is_classification:
                label_n = label_names[np.int64(predicts_cpu[rand_idx[0]])]
            label_correct.append(label_n)
            num_save_correct += 1
        else:
            images_incorrect.append((images[rand_idx[0]].cpu()))
            label_n = f'T:{np.int64(predicts_cpu[rand_idx[0]]) } - F:{np.int64(targets_cpu[rand_idx[0]])}'
            if is_classification:
                label_n = f'T:{label_names[np.int64(predicts_cpu[rand_idx[0]])]} - F:{label_names[np.int64(targets_cpu[rand_idx[0]])]}'
            label_incorrect.append(label_n)
            if np.int64(predicts_cpu[rand_idx[0]]) - np.int64(targets_cpu[rand_idx[0]]) >= 5:
                idx_worst_case.append(num_save_incorrect )

            num_save_incorrect += 1
            


    print('[-] Test loss {:.6f}'.format(np.mean(val_losses)))

    results = {
        'losses': val_losses,
        'images_correct': images_correct,
        'label_correct': label_correct,
        'images_incorrect': images_incorrect,
        'label_incorrect': label_incorrect,
        'idx_worst_case': idx_worst_case
    }

    return results, y_pred, y_true

def scores(y_true, y_pred, is_classification=True): 
    matrix = None
    f1 = None
    precision = None
    accuracy = None
    matrix = None
    roc_auc = None
    roc_plot = None
    error = None

    if is_classification:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred,)
        roc_plot = roc_curve(y_true, y_pred)
        error = 1 - accuracy

        scores_ensemble = {
            'precision': precision,
            'error': error,
            'f1-score': f1,
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy,
            'roc_auc_score': roc_auc
        }
    else:
        scores_ensemble = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'error': r2_score(y_true, y_pred),
        }

    scores_ensemble = pd.DataFrame(scores_ensemble, index=[0])

    return matrix, scores_ensemble, roc_plot