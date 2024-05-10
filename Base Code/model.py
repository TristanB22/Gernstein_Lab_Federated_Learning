# this file is meant to give the structure for the model that all of the clients and the central server are using 
# since it should be a shared structure to allow for the gradient computation


import torch
from torch import nn


# define the neural network
class HealthNet(nn.Module):
    
	def __init__(self):
		
		# init the parent object
		super(HealthNet, self).__init__()

		# define the sequential network that we are working with
		self.network = nn.Sequential(
			nn.Linear(30, 16), 
			nn.ReLU(),
			nn.Linear(16, 8),
			nn.ReLU(),
			nn.Linear(8, 1),
			nn.Sigmoid()
		)

	# define the forward pass of the network
	def forward(self, x):
		return self.fc(x)
