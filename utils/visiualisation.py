import matplotlib.pyplot as plt
import numpy as np
from utils.dataloader import postprocess

def display_graph_learning(losses_training, losses_validations, graph_title):
    assert(len(losses_training) == len(losses_validations)), 'Size losses training must be same losses validations'

    x = np.arange(0, len(losses_training))

    plt.plot(x, losses_training, color='blue', label='Perte en entrainement')
    plt.plot(x, losses_validations, color='red', label='Perte en test')
    plt.title(graph_title)

    plt.xlabel("Epochs d'entrainement")
    plt.ylabel("Perte")

    plt.legend()
    
    plt.show()

def grid_image_data_set(images, labels, graph_title):
    assert(len(images) == 9), f'Size must be 6 and got {len(images)}'

    fig, ax = plt.subplots(3, 3, tight_layout=True)
    index = 0

    for i in range(3):
        for j in range(3):
            img = images[index]
            postprocess_apply = postprocess()
            ax[i,j].imshow(postprocess_apply(img))
            ax[i,j].set_title(labels[index])
            ax[i,j].set_axis_off()
            index += 1
    
    plt.title(graph_title)
    plt.show()