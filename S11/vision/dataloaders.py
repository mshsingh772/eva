import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

class Loader:
  def __init__(self, train_transforms, test_transforms, dataset_name:str, train=True, test=True, batch_size=32, shuffle=False, num_workers=2, download=True):
    self.train_transforms = train_transforms
    self.test_transforms = test_transforms
    self.train = train
    self.test = test
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.download = download
    self.dataset_name = dataset_name

  def dataloader_train(self):
      use_cuda = torch.cuda.is_available()
      if self.dataset_name.lower() == 'cifar10': 
        trainset = torchvision.datasets.CIFAR10(root='./data', train=self.train,
                                                download=self.download, transform=self.train_transforms)
      elif self.dataset_name.lower() == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=self.train,
                                                download=self.download, transform=self.train_transforms)
      elif not len(self.dataset_name.strip()):
        print('The value for dataset_name has to be either "mnist" or "cifar10". ')                                            

      dataloader_args = dict(batch_size = self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers)  if use_cuda else dict(batch_size = self.batch_size, shuffle = self.shuffle)
      trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
      return trainloader


  def dataloader_test(self):
      use_cuda = torch.cuda.is_available()
      if self.dataset_name.lower() == 'cifar10': 
        testset = torchvision.datasets.CIFAR10(root='./data', train=not self.test,
                                                download=self.download, transform=self.test_transforms)
      elif self.dataset_name.lower() == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=not self.test,
                                                download=self.download, transform=self.test_transforms)
      elif not len(self.dataset_name.strip()):
        print('The value for dataset_name has to be either "mnist" or "cifar10". ')  
      dataloader_args = dict(batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)  if use_cuda else dict(batch_size=self.batch_size, shuffle=self.shuffle)
      testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
      return testloader


  def dataloader_gradcam(self):
    use_cuda = torch.cuda.is_available()
    if self.dataset_name.lower() == 'cifar10': 
      testset = torchvision.datasets.CIFAR10(root='./data', train=not self.test,
                                              download=self.download, transform=self.test_transforms)
    elif self.dataset_name.lower() == 'mnist':
      testset = torchvision.datasets.MNIST(root='./data', train=not self.test,
                                              download=self.download, transform=self.test_transforms)
    elif not len(self.dataset_name.strip()):
      print('The value for dataset_name has to be either "mnist" or "cifar10". ')  
    dataloader_args = dict(batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers)  if use_cuda else dict(batch_size=self.batch_size, shuffle=self.shuffle)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    return testloader

