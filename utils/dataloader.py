import torch
import os
import requests
import zipfile

from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm

URL_UTK_FACE_PART_1 = {'id': '1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW'}
URL_LANDMARK_PART_1 = {'id': '0BxYys69jI14kSi1abVV0YWFLWTg'}

URL_UTK_FACE_PART_2 = {'id': '19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b'}
URL_LANDMARK_PART_2 = {'id': '0BxYys69jI14kQlZhMVZuYnBWdUk'}

URL_UTK_FACE_PART_3 = {'id': '1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b'}
URL_LANDMARK_PART_3 = {'id': '0BxYys69jI14kLWoyYXVJTGNkWkE'}

URL_FACIAL_AGE = 'https://www.kaggle.com/datasets/frabbisw/facial-age/download?datasetVersionNumber=1'


class FaceDataset(Dataset):
    
    def __augmentation_dataset(self):
        pass
    
    def __extract_dataset(self):
        return ''
    
    def __download_dataset(self, url, name, is_google_drive=False):
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
            res = requests.get(url, stream=True)
            assert(res.status_code == 200), f'ERROR: impossible to download the image (code={res.status_code})'


        with open(f'data/{name}', 'wb') as file:
            for data in res.iter_content(chunk_size=128):
                file.write(data)
        return f'data/{name}'

    def __init_dataset(self):
        pass

    def __init__(self):
        super().__init__()
    
    def __len__(self):
        return 0
    
    def __getitem__(self, index) :
        return super().__getitem__(index)