from torchvision import  transforms
from albumentations.pytorch import ToTensor
import albumentations as A
import numpy as np
import  cv2
from PIL import Image
from torch.utils.data import Dataset
from vision.utils import Helper

class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        # Read an image with OpenCV
        image = cv2.imread(file_path)
        
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


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


def trainset_albumentations(train_dataset_x, train_y):

  ch_means = (0.48043839, 0.44820218, 0.39760034)
  ch_std = (0.27698959, 0.26908774, 0.28216029)

  train_albumentations_transform = A.Compose([
      A.Rotate((-20.0, 20.0)),
      # A.CourseDropout(0.2),
      A.HorizontalFlip(),
      A.ChannelDropout(channel_drop_range=(1, 1)),
      A.RandomBrightness(0.2),
      A.Normalize(
          mean=[0.49139968, 0.48215841, 0.44653091],
          std=[0.24703223, 0.24348513, 0.26158784],
      ),
      A.Cutout(num_holes=1, max_h_size=10, max_w_size=10, always_apply=True),
      ToTensor()
  ])
  return AlbumentationsDataset(
      file_paths=train_dataset_x,
      labels=train_y,
      transform=train_albumentations_transform,
)


def testset_albumentations(test_dataset_x, test_y):

  test_albumentations_transform = A.Compose([
      A.Normalize(
          mean=[0.49139968, 0.48215841, 0.44653091],
          std=[0.24703223, 0.24348513, 0.26158784],
      ),
      ToTensor()
  ])
  return AlbumentationsDataset(
      file_paths=test_dataset_x,
      labels=test_y,
      transform=test_albumentations_transform,
  )























