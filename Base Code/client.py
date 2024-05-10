# this is the definition file for a single client in the federated learning environment
# which is going to recieve the model weights, compute the gradient with respect to the local update
# that it has, and then return the gradient before recieving the global gradient update

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from model import HealthNet


# defining the client class 
class Client:

	def __init__(self, dataset, batch_size=16, learning_rate=0.01, mu=0.01):
		
		# the model that we are using
		self.model = HealthNet()

		# the data loader for the data that this client has access to 
		self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# optimizer for the local client
		self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

		# the loss that we are using
		self.loss_function = torch.nn.BCELoss()

		# proximal term coefficient
		self.mu = mu  

		# to hold the global model weights for fedProx
		self.global_model_weights = None


	# takes the global weights that are passed to the model
	# and updates the model state dictionary to use those weights
	def set_weights(self, global_weights):

		# set the model weights param
		self.global_model_weights = global_weights

		# load the state dictionary to the model
		self.model.load_state_dict(global_weights)

	# training with the fedprox requirement to make sure that the parameters don't move too far one way or another
	def train_with_proximal_term(self, epochs=1, use_fed_prox=True):

		# set the model to training mode
		self.model.train()
		total_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
		count_batches = 0

		# train for epochs
		for epoch in range(epochs):

			# get the data
			for data, labels in self.dataloader:

				# reset the optimizer
				self.optimizer.zero_grad()

				# forward pass
				outputs = self.model(data.float())

				# reshape for the loss function
				labels = labels.float().view(-1, 1)  
				loss = self.criterion(outputs, labels)

				# apply the fed prox to the weights
				if self.global_model_weights and use_fed_prox:
					for param, global_param in zip(self.model.parameters(), self.global_model_weights.values()):
						loss += (self.mu / 2) * torch.norm(param - global_param) ** 2

				loss.backward()
				self.optimizer.step()

				# accumulate gradients over all batches
				for name, param in self.model.named_parameters():
					total_gradients[name] += param.grad.data.clone()

				# keep track of the number of batches that we have been through so far
				count_batches += 1

		averaged_gradients = {name: grad / count_batches for name, grad in total_gradients.items()}
		return averaged_gradients, self.dataset_size


	# get the gradients of the model after training
	def get_gradients(self):
		return self.train_with_proximal_term()

	# get the weights of the client network
	def get_weights(self):
		return self.model.state_dict()
