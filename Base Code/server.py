# this is the central server that is going to aggregate all of the parameter updates from 
# each of the neural network clients that we are interacting with
import torch
from model import HealthNet


class Server:

	# initializing the server class that is going to aggregate all of the model updates and update the global model
	def __init__(self):
		
		# instantiate the model
		self.global_model = HealthNet()
		self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)

	# apply the gradients to the global model
	def apply_gradients(self, aggregated_gradients):
		with torch.no_grad():
			for name, param in self.global_model.named_parameters():
				param -= aggregated_gradients[name]

	# get the average of all of the weights to update the global model
	# this is a less efficient process than the gradient updates since there could be more 
	# information that gets passed
	def federated_avg(self, client_information):
		
		# get the total size of the gradients
		total_size = sum(size for _, size in client_information)
		aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.global_model.named_parameters()}

		# add each of the gradients using a weighted average of the data
		# so that we weight the health clients with more data more in the training
		for gradients, size in client_information:
			for name, grad in gradients.items():
				aggregated_gradients[name] += grad * (size / total_size)

		self.apply_gradients(aggregated_gradients)
		

	# get the information for the global model
	def get_global_model(self):
		return self.global_model

	# update all of the information in the global model using the method that we want to
	def update_global_model(self):
		pass 