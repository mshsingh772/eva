import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from .utils import Helper


class RangeTest():

  def __init__(self, model, epoch, max_lr, min_lr, criterion, train_dataloader):
    self.model = model
    self.epoch = epoch
    self.max_lr = max_lr
    self.min_lr = min_lr
    helper = Helper()
    self.device = helper.get_device()
    self.criterion = criterion
    self.train_dataloader = train_dataloader
    self.Lrtest_train_acc = []
    self.LRtest_Lr = []

  def lr_range_test(self, momemtum = 0.9, weight_decay=0.05):
      step = (self.max_lr - self.min_lr )/self.epoch
      lr = self.min_lr
      for ep in range(1, self.epoch+1):
          testmodel = copy.deepcopy(self.model)
          optimizer = optim.SGD(testmodel.parameters(), lr=lr, momentum=momemtum, weight_decay=weight_decay) 
          lr += (self.max_lr - self.min_lr)/self.epoch
          testmodel.train()
          pbar = tqdm(self.train_dataloader)
          correct = 0
          processed = 0
          for batch_idx, (data, target) in enumerate(pbar):
              data, target = data.to(self.device), target.to(self.device)
              optimizer.zero_grad()
              y_pred =testmodel(data)
              loss = self.criterion(y_pred, target)
              loss.backward()
              optimizer.step()
              pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()
              processed += len(data)
              pbar.set_description(desc= f'epoch = {ep} Lr = {optimizer.param_groups[0]["lr"]}  Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
          self.Lrtest_train_acc.append(100*correct/processed)
          self.LRtest_Lr.append(optimizer.param_groups[0]['lr'])

      plt.plot(self.LRtest_Lr, self.Lrtest_train_acc)
      plt.ylabel('Training Accuracy')
      plt.xlabel("Learning Rate")
      plt.title("Learning Rate v/s Accuracy")
      plt.show()
      max_y = max(self.Lrtest_train_acc)  # Find the maximum y value
      max_x = self.LRtest_Lr[self.Lrtest_train_acc.index(max_y)]  # Find the x value corresponding to the maximum y value
      # print(f'Maximum accuracy is {max_y} for the learning rate: ', max_x)
      return max_x


