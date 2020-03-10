import torch
from torchsummary import summary


def model_summary(Net, input_size):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = Net.to(device)
  summary(model, input_size=input_size)


