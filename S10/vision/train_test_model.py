import torch
from tqdm import tqdm

class RunModel:
  def __init__(self, model, trainloader, testloader, criterion, optimizer, scheduler, epochs):
    self.model = model
    self.trainloader = trainloader
    self.testloader = testloader
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.epochs = epochs
    self.train_losses = []
    self.test_losses = []
    self.train_accuracies = []
    self.test_accuracies = []

  def train(self, epoch):  
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = self.model.to(device)
      running_loss = 0.0
      pbar = tqdm(self.trainloader)
      correct = 0
      processed = 0
      train_acc = []
      for batch_idx, (inputs, labels) in enumerate(pbar):
          # get the inputs
          inputs, labels = inputs.to(device), labels.to(device)

          # zero the parameter gradients
          self.optimizer.zero_grad()
          
          # forward + backward + optimize
          outputs = model(inputs)
          loss = self.criterion(outputs, labels)
          self.train_losses.append(loss)
          loss.backward()
          self.optimizer.step()
          running_loss += loss.item()
          
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(labels.view_as(pred)).sum().item()
          processed += len(inputs)
          pbar.set_description(desc= f'Epoch: {epoch}  Loss={loss.item()}  Batch_id={batch_idx}  Train Accuracy={100*correct/processed:0.2f}')
          acc = 100*correct/processed
          train_acc.append(acc)
          self.scheduler.step(running_loss)
      self.train_accuracies.append(train_acc[-1])


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
      print('Accuracy of the network on the 10000 test images: %0.2f %% \n' % (100 * correct / total))
      acc = 100*correct/total
      self.test_accuracies.append(acc)


  def train_test(self):
      for epoch in range(1, self.epochs+1):
          self.train(epoch)
          self.test()

  def get_losses(self):
    return self.train_losses, self.test_losses
    
            
  def get_accuracies(self):
      return self.train_accuracies, self.test_accuracies


