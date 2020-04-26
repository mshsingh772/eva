from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
import requests
from sklearn.model_selection import train_test_split
from io import StringIO,BytesIO


def TinyImageNetDataSet(imagenet_path, test_size):
  dataset, class_names = compile_data(imagenet_path)
  train_dataset_x, test_dataset_x, train_y, test_y  = train_test_split([_[0] for _ in dataset],[_[1] for _ in dataset],test_size=test_size, random_state=42)
  return train_dataset_x, test_dataset_x, train_y, test_y

def download_images():
    url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    if (os.path.isdir("tiny-imagenet-200")):
        print ('Images already downloaded...')
        return
    r = requests.get(url, stream=True)
    print ('Downloading TinyImageNet Data' )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
      zip_ref.extract(member = file)
    zip_ref.close()


def compile_data(path):
    images = []
    labels = []

    class_ids = [line.strip() for line in open(os.path.join(path , 'wnids.txt'), 'r')]
    id_dict = {x:i for i, x in enumerate(class_ids)}
    all_classes = {line.split('\t')[0] : line.split('\t')[1].strip() for line in open( os.path.join(path, 'words.txt'), 'r')}
    class_names = [all_classes[x] for x in class_ids]

    # train data
    for value, key in enumerate(class_ids):
        img_path = os.path.join(path, "train", key, "images")
        images += [ os.path.join(img_path ,f"{key}_{i}.JPEG") for i in range(500)]
        labels += [value for i in range(500)]

    # validation data
    for line in open( os.path.join(path ,'val', 'val_annotations.txt')):
        img_name, class_id = line.split('\t')[:2]
        img_path = os.path.join(path, "val","images")
        images.append(os.path.join(img_path, f'{img_name}'))
        labels.append(id_dict[class_id])

    dataset = list(zip(images, labels))
    return dataset, class_names






























































































    