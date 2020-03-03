import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

def train(model, trainloader, optimizer, criterion, epoch):  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    running_loss = 0.0
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # running_loss += loss.item()
        
        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        processed += len(inputs)
        
        pbar.set_description(desc= f'Epoch: {epoch}  Loss={loss.item()}  Batch_id={batch_idx}  Accuracy={100*correct/processed:0.2f}')

def test(model, testloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %% \n' % (100 * correct / total))

def run_model(model, trainloader, testloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1, epochs+1):
        train(model, trainloader, optimizer, criterion, epoch)
        test(model, testloader)
