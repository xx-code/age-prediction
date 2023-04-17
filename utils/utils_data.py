
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset
from numpy.random import RandomState

def train_validation_test_split(dataset:Dataset, validation_size=0, test_size=0):
    rand = RandomState(seed=0)

    dsize = len(dataset)
    indices_dataset = np.arange(0, dsize, step=1)
    rand.shuffle(indices_dataset)

    split = int(np.floor(test_size*dsize))

    indices_train, indices_test = indices_dataset[split:], indices_dataset[:split]

    test_sampler = SubsetRandomSampler(indices_test)

    split_val= int(np.floor(validation_size*len(indices_train)))

    rand.shuffle(indices_train)

    indices_train, indices_validation = indices_train[split_val:], indices_train[:split_val]

    train_sampler = SubsetRandomSampler(indices_train)
    validation_sampler = SubsetRandomSampler(indices_validation)

    return train_sampler, validation_sampler, test_sampler
