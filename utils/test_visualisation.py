from visiualisation import *
from dataloader  import FaceDataset, preprocess

if __name__ == '__main__':

    fake_losses_train = [416, 488, 341, 320, 300, 278, 280, 281, 210, 174, 165, 100]
    fake_losses_test = [478, 490, 355, 370, 350, 290, 287, 260, 240, 190, 182, 120]

    display_graph_learning(fake_losses_train, fake_losses_test, "Title graph")

    faceDataset = FaceDataset(preprocess(), is_classification=True)

    label_names = faceDataset.get_all_age_range()

    images = []
    labels = []

    for i in range(9):
        rand_idx = np.random.randint(0, len(faceDataset))
        img, label = faceDataset[rand_idx]

        images.append(img)
        labels.append(label)

    grid_image_data_set(images, labels, "")