import torch
from tqdm import tqdm
import torch.nn.functional as F

class RunModel:
  def __init__(self, model, trainloader, testloader, optimizer, scheduler, epochs, criterion, L1=0):
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
    self.L1 = L1

  def train(self): 
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = self.model.to(device)
      model.train() 
      pbar = tqdm(self.trainloader)
      correct = 0
      processed = 0
      loss = 0

      for batch_idx, (inputs, labels) in enumerate(pbar):
          # get the inputs
          inputs, labels = inputs.to(device), labels.to(device)

          # zero the parameter gradients
          self.optimizer.zero_grad()
          
          # forward + backward + optimize
          
          outputs = model(inputs)
          loss = self.criterion(outputs, labels)

          #Implementing L1 regularization
          if self.L1 > 0:
            reg_loss = 0.
            for param in model.parameters():
              reg_loss += torch.sum(param.abs())
            loss += self.L1 * reg_loss
            
          loss.backward()
          self.optimizer.step()
          
          pred = outputs.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
          correct += pred.eq(labels).sum().item()
          processed += len(inputs)
          
          pbar.set_description(desc=f' Loss={loss.item()} Train Accuracy={(100*correct/processed):.2f}%')
          pbar.update(1)
      
      self.scheduler.step(loss)
      self.train_accuracies.append(100*correct/processed)
      self.train_losses.append(loss)


  def test(self):
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = self.model.to(device)
      model.eval()
      correct = 0
      total = 0
      test_loss = 0
      with torch.no_grad():
          for images, labels in self.testloader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              test_loss += self.criterion(outputs, labels).item()
              pred = outputs.argmax(dim=1, keepdim=False) 
              correct += pred.eq(labels).sum().item()
              
      test_loss /= len(self.testloader.dataset)
      self.test_losses.append(test_loss)
      self.test_accuracies.append(100*correct/len(self.testloader.dataset))
      print(f'Testing: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.testloader.dataset)} ({self.test_accuracies[-1]:.2f}%)\n')
      

  def train_test(self):
      for epoch in range(1, self.epochs+1):
          print(f'\nEpoch {epoch}:')
          print('---------')
          self.train()
          self.test()


  def get_losses(self):
    return self.train_losses, self.test_losses
    
            
  def get_accuracies(self):
      return self.train_accuracies, self.test_accuracies
  
