import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import Image, display

class Res18(nn.Module):
    def __init__(self, net):
        super(Res18, self).__init__()
        
        self.res18 = net
        self.features_conv = nn.Sequential(self.res18.conv1,
                                           self.res18.bn1,
                                           self.res18.layer1,
                                           self.res18.layer2,
                                           self.res18.layer3,
                                           self.res18.layer4
                                           ) 
        
        self.linear = self.res18.linear
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

def getheatmap(pred, class_pred, netx, img):
  pred[:, class_pred].backward()
  gradients = netx.get_activations_gradient()
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = netx.get_activations(img.cuda()).detach()
  for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = np.maximum(heatmap.cpu(), 0)
  heatmap /= torch.max(heatmap)
  return heatmap

def imshow(img, ax):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

def superposeimage(heatmap, img):
  heat1 = np.array(heatmap)
  heatmap1 = cv2.resize(heat1, (img.shape[1], img.shape[0]))
  heatmap1 = np.uint8(255 * heatmap1)
  heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
  superimposed_img = heatmap1 * 0.4 + img
  cv2.imwrite('./map.jpg', superimposed_img)

def get_gradcam(net, img, classes, gt, pd):
  netx = Res18(net)
  netx.eval()
  fig, axes = plt.subplots(nrows=1, ncols=3)
  pred = netx(img.cuda())
  from torchvision.utils import save_image
  imx = img[0]
  save_image(imx, 'img1.png')
  class_pred = int(np.array(pred.cpu().argmax(dim=1)))
  imshow(torchvision.utils.make_grid(img),axes[0])
  heatmap = getheatmap(pred, class_pred, netx, img)
  axes[1].matshow(heatmap.squeeze())
  axes[0].set_title("Actual: "+gt)
  axes[2].set_title("Predicted: "+pd)
  imx = cv2.imread("./img1.png")
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
  superposeimage(heatmap, imx)
  imx = cv2.imread('./map.jpg')
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
  axes[2].imshow(imx, cmap='gray', interpolation='bicubic')
