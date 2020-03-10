import torch
from tqdm import tqdm

class RunModel:
  def __init__(self, model, trainloader, testloader, criterion, optimizer, epochs):
    self.model = model
    self.trainloader = trainloader
    self.testloader = testloader
    self.criterion = criterion
    self.optimizer = optimizer
    self.epochs = epochs

  def train(self, epoch):  
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = self.model.to(device)
      running_loss = 0.0
      pbar = tqdm(self.trainloader)
      correct = 0
      processed = 0
      for batch_idx, (inputs, labels) in enumerate(pbar):
          # get the inputs
          inputs, labels = inputs.to(device), labels.to(device)

          # zero the parameter gradients
          self.optimizer.zero_grad()
          
          # forward + backward + optimize
          outputs = model(inputs)
          loss = self.criterion(outputs, labels)
          loss.backward()
          self.optimizer.step()
          # running_loss += loss.item()
          
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(labels.view_as(pred)).sum().item()
          processed += len(inputs)
          pbar.set_description(desc= f'Epoch: {epoch}  Loss={loss.item()}  Batch_id={batch_idx}  Train Accuracy={100*correct/processed:0.2f}')

  def test(self):
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      correct = 0
      total = 0
      with torch.no_grad():
          for images, labels in self.testloader:
              images, labels = images.to(device), labels.to(device)
              outputs = self.model(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      print('Accuracy of the network on the 10000 test images: %d %% \n' % (100 * correct / total))

  def train_test(self):
      for epoch in range(1, self.epochs+1):
          self.train(epoch)
          self.test()
