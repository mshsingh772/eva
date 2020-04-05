import torch
from tqdm import tqdm
import torch.nn.functional as F

class RunModel:
  def __init__(self, model, trainloader, testloader, optimizer, scheduler, epochs, L1=0, criterion=None):
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

  def train(self, epoch):  
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = self.model.to(device)
      running_loss = 0.0
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
          if self.criterion:
            loss = self.criterion(outputs, labels)
          else:            
            loss = F.nll_loss(outputs, labels)
          

          #Implementing L1 regularization
          if self.L1 > 0:
            reg_loss = 0.
            for param in model.parameters():
              reg_loss += torch.sum(param.abs())
            loss += self.L1 * reg_loss
            
          loss.backward()
          self.optimizer.step()
          running_loss += loss.item()
          
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(labels.view_as(pred)).sum().item()
          processed += len(inputs)

          pbar.set_description(desc= f'Epoch: {epoch}  Loss={loss.item()}  Batch_id={batch_idx}  Train Accuracy={100*correct/processed:0.2f}')
          acc = 100*correct/processed

          
          self.scheduler.step(running_loss)
      self.train_accuracies.append(100*correct/processed)
      self.train_losses.append(loss)


  def test(self):
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      correct = 0
      total = 0
      test_loss = 0
      with torch.no_grad():
          for images, labels in self.testloader:
              images, labels = images.to(device), labels.to(device)
              outputs = self.model(images)
              test_loss += self.criterion(outputs, labels)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              
      
      test_loss /= len(self.testloader.dataset)
      self.test_losses.append(test_loss)
      print('Accuracy of the network on 10000 the test images: %0.2f %% \n' % (100 * correct / total))
      self.test_accuracies.append(100. * correct / len(self.testloader.dataset))


  def train_test(self):
      for epoch in range(1, self.epochs+1):
          self.train(epoch)
          self.test()

  def get_losses(self):
    return self.train_losses, self.test_losses
    
            
  def get_accuracies(self):
      return self.train_accuracies, self.test_accuracies
  
def train(model, train_loader, device, optimizer, criterion):
    print('---------')
    model = model.to(device)
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()
        processed += len(data)
        pbar.set_description(desc=f'Loss={loss.item():0.2f} Accuracy={(100 * correct / processed):.2f}')
        pbar.update(1)
    train_acc = 100*correct/processed
    return train_acc, loss

def val(model, val_loader, device, criterion, losses, accuracies):
    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            img_batch = data  
            data, target = data.to(device), target.to(device)  
            output = model(data)  
            val_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=False) 

            correct += pred.eq(target).sum().item()
    
    val_loss /= len(val_loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(val_loader.dataset))
    print(f'Testing: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracies[-1]:.2f}%)\n')
    test_acc = (100 * correct / len(val_loader.dataset))  

    return test_acc, val_loss


