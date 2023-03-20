import unittest
import os

from utils.dataloader import FaceDataset

URL = {'id': '1lBfNh2BG5excU1e2hiFUoW2tgoZtv__W'}

class TestClassFaceDataset(unittest.TestCase):

    def test_download_file(self):
        faceDataset = FaceDataset()
        path_file = faceDataset._FaceDataset__download_dataset(URL, 'test.zip', True)

        self.assertTrue(os.path.exists('data')) 
        self.assertTrue(len(path_file)>= 0)
        self.assertTrue(len(os.listdir('data')) > 0)

        os.remove(path_file)
        os.rmdir('data')

    def test_extration_file(self):
        faceDataset = FaceDataset()
        path_file = faceDataset._FaceDataset__download_dataset(URL, 'test.zip', True)
        list_path_file = faceDataset._FaceDataset__extract_dataset(path_file, False)

        self.assertTrue(os.path.exists('data')) 
        self.assertTrue(len(list_path_file)> 0)
        self.assertTrue(len(os.listdir('data')) > 1)

        os.remove(path_file)
        for path in list_path_file:
            os.remove(path)
        os.rmdir('data/images')
        os.rmdir('data')
    
    def test_age_range(self):
        faceDataset = FaceDataset()

        req_age = faceDataset.get_age_range(1)
        self.assertEqual(req_age, 0)

        req_age = faceDataset.get_age_range(24)
        self.assertEqual(req_age, 4)

        req_age = faceDataset.get_age_range(100)
        self.assertEqual(req_age, 20)

        req_age = faceDataset.get_age_range(132)
        self.assertEqual(req_age, 20)

        print(faceDataset.get_all_age_range())

        


if __name__=='__main__':
    unittest.main()