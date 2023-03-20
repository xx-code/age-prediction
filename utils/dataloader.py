import torch
import os
import shutil
import requests
import zipfile
import torchvision
import tarfile
import csv
import math
import pandas


from torch.utils.data import Dataset
from tqdm import tqdm

URL_UTK_FACE_PART_1 = {'id': '0BxYys69jI14kRjNmM0gyVWM2bHM'}
URL_UTK_FACE_PART_2 = {'id': '0BxYys69jI14kYVM3aVhKS1VhRUk'}
URL_FACIAL_AGE = {'id': '18JDhu3egCTBfmHdgoPkLos1dkVLxDaoB'}

RANGE_YEARS=5

def dataAugmentation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomRotation((-30,30)),
        ]
    )


class FaceDataset(Dataset):
    def get_age_range(self, age):
        if age >= 100:
            return 100/RANGE_YEARS
        return int(math.floor(age/RANGE_YEARS) ) 

    def get_all_age_range(self):
        age_range = []
        for i in range(0, 100, RANGE_YEARS):
            rng = f'{i} - {i+RANGE_YEARS-1}'
            age_range.append(rng)
        age_range.append(f'+100')

        return dict((i, age_rng) for i, age_rng in enumerate(age_range))


    def __extract_dataset(self, pathfile, is_tar=True):
        print('- Extract Data')
        if not os.path.exists('data'):
            os.mkdir('data')

        if not os.path.exists('data/images'):
            os.mkdir('data/images')

        pathfiles = []
        if not is_tar:
            with zipfile.ZipFile(pathfile, 'r') as filezip:
                tqdm(filezip.extractall('data'))
        else:
            with tarfile.open(pathfile, 'r') as filetar:
                tqdm(filetar.extractall('data'))

        list_path = os.listdir('data')

        
        for path in list_path:
            if os.path.isdir('data/'+path) and path != 'images':
                pathfiles.append(f'data/images/{path}')
                shutil.move(f'data/{path}', 'data/images')
            
        print('- Done !')

        return pathfiles
    
    def __download_dataset(self, url, name, is_google_drive=False, is_txt=False):
        if not os.path.exists('data'):
            os.mkdir('data')

        res = None

        if is_google_drive:
            URL = "https://drive.google.com/uc?"
            session = requests.Session()
            
            params = {'id': url['id'], 'alt' : 'media'}
            res = session.get(URL, params=params, stream=True)
            assert(res.status_code == 200), f'ERROR: impossible to download the image (code={res.status_code})'

            params = {'id': url['id'], 'confirm' : 'download_warning' }
            res = session.get(URL, params = params, stream = True)
        else:
            session = requests.Session()

            params = {'datasetVersionNumber': '1'}
            res = session.get(url, params=params, stream=True)
            assert(res.status_code == 200), f'ERROR: impossible to download the image (code={res.status_code})'

        open_mode = 'wb'
        if is_txt:
            open_mode = 'w'

        with open(f'data/{name}', open_mode) as file:
            for data in tqdm(res.iter_content(chunk_size=128)):
                file.write(data)
        return f'data/{name}'

    def __init_dataset(self, is_face_age=False, is_utk1=False, is_utk2=False):
        if not is_face_age:
            print('- Download data face age kaggle')
            path_dataset_fa = self.__download_dataset(URL_FACIAL_AGE, "face_age.zip", is_google_drive=True)
            print("")
            print('- Done!')
        else:
            path_dataset_fa = 'data/face_age.zip'

        print('- Download data UTK face')
        if not is_utk1:
            path_dataset_utk1 = self.__download_dataset(URL_UTK_FACE_PART_1, 'utk1.tar.gz', is_google_drive=True)
        else:
            path_dataset_utk1 = 'data/utk1.tar.gz'
        
        if not is_utk2:
            path_dataset_utk2 = self.__download_dataset(URL_UTK_FACE_PART_2, 'utk2.tar.gz', is_google_drive=True)
        else:
            path_dataset_utk2 = 'data/utk2.tar.gz'

        print("")
        print('- Download data Done!')

        count_img = 0

        print('- Arrange Dataset')
        paths_extract_fa = self.__extract_dataset(path_dataset_fa, is_tar=False)
        # we get face_age folder
        # for each folder  folder in data/images/face_age
        sub_dir = []
        for src_path in tqdm(os.listdir(paths_extract_fa[0])):
            # Example data/images/face_age/001; ../../face_age/002;...
            sub_dir.append(f'{paths_extract_fa[0]}/{src_path}')
        
        # loop in folder rename file image with prefix=name_folder; data/images/folder/[folder]_img[iter].jpg
        for sub_path in tqdm(sub_dir):
            age = int(sub_path[len(sub_path)-3:len(sub_path)])
            for path in os.listdir(sub_path):
                os.renames(f'{sub_path}/{path}', f'{sub_path}/{age}_img{count_img}.jpg')
                shutil.move(f'{sub_path}/{age}_img{count_img}.jpg', f'data/images/{age}_img{count_img}.jpg')
                count_img += 1
            os.rmdir(sub_path)
        os.rmdir(paths_extract_fa[0])

        # remove in memory
        del sub_dir

        
        paths_extract_utk1 = self.__extract_dataset(path_dataset_utk1)
        # we get crap_part1 folder
        # for each folder  folder in data/images/crap_part1
        # rename file image with prefix=name_folder; data/images/[folder]_img[iter].jpg
        for path in tqdm(os.listdir(paths_extract_utk1[0])):
            s_idx = path.find('_')
            age = int(path[:s_idx])

            os.rename(f'{paths_extract_utk1[0]}/{path}', f'{paths_extract_utk1[0]}/{age}_img{count_img}.jpg')
            shutil.move(f'{paths_extract_utk1[0]}/{age}_img{count_img}.jpg', f'data/images/{age}_img{count_img}.jpg')
            count_img += 1

        os.rmdir(paths_extract_utk1[0])

        paths_extract_utk2 = self.__extract_dataset(path_dataset_utk2)
        for path in tqdm(os.listdir(paths_extract_utk2[0])):
            s_idx = path.find('_')
            age = int(path[:s_idx])

            os.rename(f'{paths_extract_utk2[0]}/{path}', f'{paths_extract_utk2[0]}/{age}_img{count_img}.jpg')
            shutil.move(f'{paths_extract_utk2[0]}/{age}_img{count_img}.jpg', f'data/images/{age}_img{count_img}.jpg')
            count_img += 1
            
        os.rmdir(paths_extract_utk2[0])
        

        # create anotation csv for image and label
        with open('data/anotations.csv', 'w', newline='') as csvfile:
            fieldnames = ['path_img', 'age', 'age_range']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for path in tqdm(os.listdir('data/images')):
                s_idx = path.find('_')
                age = int(path[:s_idx])
                age_range = self.get_age_range(age)
                writer.writerow({'path_img': f'data/images/{path}', 'age': age, 'age_range': age_range})

            print('- Dataset is ready!')
                

    def __init__(self, transform=None, is_classification=False):
        if not os.path.exists('data'):
            self.__init_dataset()
        else:
            is_utk1_there = os.path.exists('data/utk1.tar.gz')
            is_utk2_there = os.path.exists('data/utk2.tar.gz')
            is_face_age_there = os.path.exists('data/face_age.zip')
            if not os.path.exists('data/images'):
                self.__init_dataset(is_face_age_there, is_utk1_there, is_utk2_there)

        
        self.transform = transform
        self.img_dir = 'data/images'
        self.img_labels = pandas.read_csv('data/anotations.csv')
        self.is_classification = is_classification
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index) :
        idx_label = 1
        if self.is_classification:
            idx_label = 2

        img = torchvision.io.read_image(self.img_labels.iloc[index, 0])

        if self.transform:
            img = self.transform(img)
        
        label = self.img_labels.iloc[index, idx_label]

        return img, label