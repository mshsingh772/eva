from torchvision import  transforms
from albumentations.pytorch import ToTensor
import albumentations as A
import numpy as np
import  cv2
from vision.utils import Helper
# from albumentations.augmentations.transforms import CoarseDropout

class TorchTransforms():

    def __init__(self, test_transforms_list, train_transforms_list= None):
        self.train_transforms_list = train_transforms_list
        self.test_transforms_list = test_transforms_list


    def trainTransform(self):
        return transforms.Compose(self.train_transforms_list)


    def testTransform(self):
        return transforms.Compose(self.test_transforms_list)


class album_transforms():

    def __init__(self):
        helper = Helper()
        self.mean, self.std = helper.get_mean_and_std('cifar10')
        # self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        # self.mean = np.array([0.4914, 0.4822, 0.4465])
        # self.std = np.array([0.2023, 0.1994, 0.2010])


        
        self.albumentation_transforms = A.Compose([
          A.HorizontalFlip(),  
          A.RandomBrightness(),
          A.Rotate((-9.0, 9.0)),
          A.Cutout(1,5,5,self.mean.mean()),
          #------------------
            # A.HueSaturationValue(p=0.25),
            # A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=15),
            # # A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_height=4,
            # #       min_width=4, fill_value=mean*255.0, p=0.75),
            # A.Cutout(num_holes=6),
            A.Normalize(
                mean=self.mean,
                std=self.std
            ), ToTensor()
            ])


    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img