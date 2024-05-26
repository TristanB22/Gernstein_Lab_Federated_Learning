# this file is meant to give the structure for the model that all of the clients and the central server are using 
# since it should be a shared structure to allow for the gradient computation


import torch
from torch import nn
from torch.nn import Dropout, BatchNorm1d


# define the neural network
class HealthNet(nn.Module):
    
	def __init__(self, input_shape):
		
		# init the parent object
		super(HealthNet, self).__init__()

		# define the sequential network that we are working with
		self.network = nn.Sequential(
			nn.Linear(input_shape, 16), 
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.BatchNorm1d(16),
			nn.Linear(16, 8),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.BatchNorm1d(8),
			nn.Linear(8, 1),
			nn.Sigmoid()
		)

	# define the forward pass of the network
	def forward(self, x):
		return self.fc(x)
