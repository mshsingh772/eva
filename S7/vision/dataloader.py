import torch
import torchvision
import torchvision.transforms as transforms

def dataloader_train(train, train_transforms, batch_size=32, shuffle=False, num_worker=2, download=True):

    use_cuda = torch.cuda.is_available()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=download, transform=train_transforms)
    dataloader_args = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)  if use_cuda else dict(batch_size=batch_size, shuffle=shuffle)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    return trainloader


def dataloader_test(test, test_transforms, batch_size=32, shuffle=False, num_worker=2, download=True):
    use_cuda = torch.cuda.is_available()
    testset = torchvision.datasets.CIFAR10(root='./data', train = not test,
                                        download=download, transform=test_transforms)
    dataloader_args = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)  if use_cuda else dict(batch_size=batch_size, shuffle=shuffle)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    return testloader