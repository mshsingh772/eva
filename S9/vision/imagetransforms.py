from torchvision import  transforms
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, RandomBrightness, Cutout, ElasticTransform
from albumentations.pytorch import ToTensor
import albumentations
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


class album_transforms():

    def __init__(self):
        helper = Helper()
        self.mean, self.std = helper.get_mean_and_std('cifar10')

        self.albumentation_transforms = Compose([
            HorizontalFlip(),  
            # ElasticTransform(p=1,border_mode=cv2.BORDER_REFLECT_101,alpha_affine=40),
            # flag that is used to specify the pixel extrapolation method.
            RandomBrightness(),
            Cutout(1,5,5,self.mean.mean()),
            # CoarseDropout(),
            Rotate((-9.0, 9.0)),
            Normalize(
                mean=self.mean,
                std=self.std
            ), ToTensor()])


    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img