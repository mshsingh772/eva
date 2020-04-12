from torchvision import  transforms
from albumentations.pytorch import ToTensor
import albumentations as A
import numpy as np
import  cv2
from vision.utils import Helper

class TorchTransforms():

    def __init__(self, test_transforms_list, train_transforms_list= None):
        self.train_transforms_list = train_transforms_list
        self.test_transforms_list = test_transforms_list


    def trainTransform(self):
        return transforms.Compose(self.train_transforms_list)


    def testTransform(self):
        return transforms.Compose(self.test_transforms_list)


class album_train_transforms():

    def __init__(self):
        helper = Helper()
        # self.mean, self.std = helper.get_mean_and_std('cifar10')
        self.mean, self.std = (0.485,0.456,0.406), (0.229,0.224,0.225)
        # self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        self.albumentation_transforms = A.Compose([
                      A.HorizontalFlip(),
                      A.Rotate((-10.0, 10.0)),
                      A.Normalize(
                        mean=self.mean,
                        std=self.std,
                      ),
                      A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, always_apply=True,value=0.5, p=1.0),
                      A.RandomCrop(height=32, width=32, always_apply=True),
                      A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=True, p=0.5),
                      ToTensor()
            ])


    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img



class album_test_transforms():

    def __init__(self):
        helper = Helper()
        # self.mean, self.std = helper.get_mean_and_std('cifar10')
        self.mean, self.std = (0.485,0.456,0.406), (0.229,0.224,0.225)
        # self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        self.albumentation_transforms = A.Compose([
          A.Normalize(
                        mean=self.mean,
                        std=self.std,
                      ),
          ToTensor(),
            ])


    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img


