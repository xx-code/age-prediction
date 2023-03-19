import unittest
import os

from utils.dataloader import FaceDataset

URL = {'id': '1lBfNh2BG5excU1e2hiFUoW2tgoZtv__W'}

class TestClassFaceDataset(unittest.TestCase):

    def test_download_file(self):
        faceDataset = FaceDataset()
        path_files = faceDataset._FaceDataset__download_dataset(URL, 'test.zip', True)

        self.assertTrue(os.path.exists('data')) 
        self.assertTrue(len(path_files)>= 0)
        self.assertTrue(len(os.listdir('data')) > 0)

        os.remove(path_files)
        os.rmdir('data')

        


if __name__=='__main__':
    unittest.main()