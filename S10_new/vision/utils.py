import torch
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import torchvision.transforms as transforms
from vision.gradcam import GradCAM
from torchvision.utils import make_grid
import  torchvision
import matplotlib.pyplot as plt

class Helper():
  def __init__(self):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    self.device = device


  def model_summary(self, Net, input_size):
    model = Net.to(self.device)
    summary(model, input_size=input_size)


  def get_mean_and_std(self, dataset_name):
    mean,std = 0,0
    train_transforms = transforms.Compose([
                                       transforms.ToTensor()
                                        ])
    if dataset_name.lower() == 'cifar10': 
        train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transforms)
        mean, std = train.data.mean(axis=(0,1,2))/255, train.data.std(axis=(0,1,2))/255

    elif dataset_name.lower() == 'mnist':
      train = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=train_transforms)
      train_data = train.train_data
      train_data = train.transform(train_data.numpy())
      mean, std = torch.mean(train_data), torch.std(train_data)

    return mean, std


  def transform_to_device(self, pil_img):
      torch_img = transforms.Compose([
          transforms.Resize((32, 32)),
          transforms.ToTensor()
      ])(pil_img).to(self.device)
      norm_torch_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(torch_img)[None]
      return torch_img, norm_torch_img


  def visualize_cam(self, mask, img, alpha=1.0):
      """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
      Args:
          mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
          img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
      Return:
          heatmap (torch.tensor): heatmap img shape of (3, H, W)
          result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
      """
      heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
      heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
      b, g, r = heatmap.split(1)
      heatmap = torch.cat([r, g, b]) * alpha

      result = heatmap+img.cpu()
      result = result.div(result.max()).squeeze()

      return heatmap, result

  def show_img(self, img):
      #img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      fig = plt.figure(figsize=(6,6))
      plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')

  def plot_images(self, torch_img,normed_torch_img, model):
      images=[]
      g1 = GradCAM(model, model.layer1)
      g2 = GradCAM(model, model.layer2)
      g3 = GradCAM(model, model.layer3)
      g4 = GradCAM(model, model.layer4)
      mask1, _ = g1(normed_torch_img)
      mask2, _ = g2(normed_torch_img)
      mask3, _ = g3(normed_torch_img)
      mask4, _ = g4(normed_torch_img)
      heatmap1, result1 = self.visualize_cam(mask1, torch_img)
      heatmap2, result2 = self.visualize_cam(mask2, torch_img)
      heatmap3, result3 = self.visualize_cam(mask3, torch_img)
      heatmap4, result4 = self.visualize_cam(mask4, torch_img)

      images.extend([torch_img.cpu(), heatmap1, heatmap2, heatmap3, heatmap4])
      images.extend([torch_img.cpu(), result1, result2, result3, result4])
      grid_image = make_grid(images, nrow=5)
      self.show_img(grid_image)
  
  
  def missclassified_classes(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    img_arr = []
    img_pred = []
    img_org = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Store wrongly predicted images
            wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
            wrong_samples = data[wrong_idx]
            wrong_preds = pred[wrong_idx]
            actual_preds = target.view_as(pred)[wrong_idx]

            for i , c in  enumerate(wrong_idx.tolist()):
                sample = wrong_samples[i].cpu()
                wrong_pred = wrong_preds[i].cpu()
                actual_pred = actual_preds[i].cpu()
                # Undo normalization
                sample = sample * 0.3081
                sample = sample + 0.1307
                sample = sample * 255.
                sample = sample.byte()
                img = TF.to_pil_image(sample)
                img_arr.append(img)
                img_pred.append(wrong_pred.item())
                img_org.append(actual_pred.item())
    
    return img_arr, img_org, img_pred

  def plot_acc_graph(self, train_accs, test_accs, epochs):
    epoch_count = range(0, epochs)
    plt.figure(1, figsize=[15,10])
    plt.plot(epoch_count, train_accs, color='r')
    plt.plot(epoch_count, test_accs, color='c')
    plt.legend(['Train Acc', 'Test Acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title("Train-Test Accuracy Graph")
    plt.show()
  
  def plot_loss_graph(self, train_losses, test_losses, epochs):
    epoch_count = range(0, epochs)
    plt.figure(1, figsize=[15,10])
    plt.plot(epoch_count, train_losses, color='r')
    plt.plot(epoch_count, test_losses, color='c')
    plt.legend(['Train Losses', 'Test Losses'])
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.title("Train-Test Loss Graph")
    plt.show()




