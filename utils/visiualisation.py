import matplotlib.pyplot as plt
import numpy as np
from utils.dataloader import postprocess

def display_graph_learning(losses_training, losses_validations, std_training, std_validations, graph_title, do_save=False, file_name=''):
    x = np.arange(0, len(losses_training))

    plt.plot(x, losses_training, color='blue', label='Perte en entrainement')
    plt.fill_between(x, losses_training, losses_training+std_training, alpha=0.4)

    if len(losses_validations) > 0:
        plt.plot(x, losses_validations, color='red', label='Perte en test')
        plt.fill_between(x, losses_validations, losses_validations+std_validations, alpha=0.4)

    plt.title(graph_title)

    plt.xlabel("Epochs d'entrainement")
    plt.ylabel("Perte")

    plt.legend()
    
    plt.show()

    if do_save == True:
        plt.savefig(f'{file_name}.jpg')

def grid_image_data_set(images, labels, graph_title, do_save=False, file_name=''):
    rows = len(images)//3
    idx_no_display = False
    if len(images)%3 != 0:
        idx_no_display = True

    fig, ax = plt.subplots(rows, 3, tight_layout=True)
    index = 0

    for i in range(rows):
        for j in range(3):
            if idx_no_display and index == len(images) - 1:
                break;
            img = images[index]
            postprocess_apply = postprocess()
            ax[i,j].imshow(postprocess_apply(img))
            ax[i,j].set_title(labels[index])
            ax[i,j].set_axis_off()
            index += 1

    fig.suptitle(graph_title, fontsize=16)
    plt.show()

    if do_save == True:
        plt.savefig(f'{file_name}.jpg')

def display_matrix_coffusion(matrix, do_save=False, file_name=''):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuel', fontsize=12)
    plt.title('Matrix de confusion', fontsize=12);

    plt.show()

    if do_save == True:
        plt.savefig(f'{file_name}.jpg')