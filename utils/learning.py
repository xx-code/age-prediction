import numpy as np
import torch

from tqdm import tqdm

def training(model, n_epoch, train_loader, optimizer, criterion, validation_loader, patient=0, DEVICE='cpu', do_save=True):
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

    for i_epoch in range(n_epoch):
        train_losses = []
        val_losses = []

        model.train()
        print(f'Training process epoch {i_epoch+1}/{n_epoch}')
        for batch in tqdm(train_loader):
            images, targets = batch

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.float()

            optimizer.zero_grad()

            predicts = model(images)
            
            predicts = torch.reshape(predicts, (predicts.size()[0], ))

            loss = criterion(predicts, targets)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
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

            val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_validation =  np.mean(val_losses)

        if patient > 0:
            if early_stopping == patient:
                stop = True

            if mean_validation < best_val_acc:
                best_val_acc = mean_validation
                early_stopping = 0
            else:
                early_stopping += 1

        print('[-] epoch {:}/{:}, train loss {:.6f}, valiation loss {:.6f}'.format(
        i_epoch+1, n_epoch, mean_train, mean_validation))
        history_train_losses.append((mean_train, np.std(train_losses)))
        history_validation_losses.append((mean_validation, np.std(val_losses)))
        print(" ")

        if stop:
            break
    
    if do_save:
        torch.save(model, 'models/restnet_custom.pt')

    return model, history_train_losses, history_validation_losses




def testing(model, criterion, test_validation, is_classification=False, DEVICE='cpu'):
    model.eval()

    progress_test_dataloader = tqdm(test_validation, desc=f"Test")

    images_correct = []
    images_incorrect = []

    val_losses = []
    num_save_correct = 0
    num_save_incorrect = 0

    y = []
    y_true = []

    for batch in progress_test_dataloader:
        images, targets = batch

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.no_grad():
            predicts = model(images)
            predicts = torch.reshape(predicts, (predicts.size()[0], ))
        
        loss = criterion(predicts, targets)
        val_losses.append(loss.item())

        rand_idx = torch.randint(0, len(targets), (1,))

        y = np.append(y, targets.cpu().numpy().flatten())
        y_true = np.append(y_true, targets.cpu().numpy().flatten())

        if targets.cpu().flatten()[rand_idx[0]] == predicts.cpu().flatten()[rand_idx[0]]:
            if num_save_correct < 9:
                images_correct.append(images[rand_idx[0]])
                num_save_correct += 1
        else:
            if num_save_incorrect < 9:
                images_incorrect.append(images[rand_idx[0]])
                num_save_incorrect += 1

    print('[-] Test loss {:.6f}'.format(np.mean(val_losses)))

    return images_correct, images_incorrect, y, y_true