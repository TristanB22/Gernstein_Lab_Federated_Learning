

# this is the central server that is going to aggregate all of the parameter updates from 
# each of the neural network clients that we are interacting with
import torch
from typing import List
from torch.utils.data import DataLoader, TensorDataset, random_split

from tqdm import tqdm

from .client import Client
from .model import HealthNet

# the server class that is going to manage all of the clients and their updates
class Server:

	# initializes the server object
	# device: the device that the server is going to be running on
	# clients: the clients that are going to be used in the training process
	# environment: the environment that the server is going to be running in
	def __init__(self, 
              device: torch.device, 
              clients: List[Client], 
              environment: str = 'central',
              num_rounds=10,
			  batch_size=32
              ):
		
		self.model = HealthNet(30).to(device)
		self.device = device
		self.clients = clients
		self.environment = environment
		self.num_rounds = num_rounds
		self.batch_size = batch_size
  
		self.client_weights = []
		self.client_gradients = []


	# function to aggregate the parameters of the clients
	def aggregate_parameters(self):
		
		with torch.no_grad():
			
			# initialize aggregated weights with the shape of model parameters
			aggregated_weights = {name: torch.zeros_like(param, device=self.device) 
								for name, param in self.model.named_parameters()}
			
			# initialize aggregated buffers (for BatchNorm parameters)
			aggregated_buffers = {name: torch.zeros_like(buffer, device=self.device)
								for name, buffer in self.model.named_buffers()}

			# sum all client parameters
			for client_state in self.client_weights:
				for name, param in client_state.items():
					if name in aggregated_weights:
						aggregated_weights[name] += param
					elif name in aggregated_buffers:
						aggregated_buffers[name] += param
      
			# divide by the number of clients
			for name in aggregated_weights:
				aggregated_weights[name] /= len(self.client_weights)
			for name in aggregated_buffers:
				aggregated_buffers[name] = aggregated_buffers[name].float() / len(self.client_weights)

			# convert aggregated weights back to long if necessary
			for name, param in aggregated_weights.items():
				if self.model.state_dict()[name].dtype == torch.long:
					aggregated_weights[name] = aggregated_weights[name].to(dtype=torch.long)
			for name, buffer in aggregated_buffers.items():
				if self.model.state_dict()[name].dtype == torch.long:
					aggregated_buffers[name] = aggregated_buffers[name].to(dtype=torch.long)

			# load the aggregated weights and buffers into the server model
			combined_state_dict = {**aggregated_weights, **aggregated_buffers}
			self.model.load_state_dict(combined_state_dict, strict=False)
   
			# clear the weights after aggregation
			self.client_weights = []
	

	# function to aggregate the gradients of the clients
	def aggregate_gradients(self):
	
		with torch.no_grad():
      
			# initialize aggregated gradients with the shape of model parameters
			aggregated_gradients = {name: torch.zeros_like(param, device=self.device)
									for name, param in self.model.named_parameters()}

			# sum all client gradients
			for client_state in self.client_gradients:
				for name, grad in client_state.items():
					aggregated_gradients[name] += grad / len(self.client_gradients)

			# apply the aggregated gradients to the server model parameters
			for name, param in self.model.named_parameters():
				param -= aggregated_gradients[name]
    
			# clear the gradients after aggregation
			self.client_gradients = []


	# define a general run function that chooses the strategy based on the environment
	def run(self, num_rounds=-1):
     
		# reset the number of rounds if it is passed in
		if num_rounds != -1:
			self.num_rounds = num_rounds
     
		# go through the number of rounds that we are going to be training for
		for _ in tqdm(range(self.num_rounds), desc='Training Rounds'):
      
			# reset the weights of the clients to be the same as the server weights
			self.reset_distributed_weights()
		
			# run the training based on the environment
			if self.environment == 'central':
				self.central_training()
			elif self.environment == 'federated':
				self.federated_training()
			elif self.environment == 'distributed':
				self.distributed_training()
			else:
				raise ValueError('Environment not recognized.')


	# function to set up the training environment for the central server
	def setup_distributed_environment(self):
     
     	# depending on the environment that we are working in do different things
		if self.environment == 'central':
			pass
		elif self.environment == 'federated':
			pass
		elif self.environment == 'distributed':	
			pass
		else:
			raise ValueError('Environment not recognized.')

	
 	# function to aggregate data from clients
	def aggregate_data_from_clients(self):
		
  		# retrieve datasets from all clients
		client_datasets = [client.get_dataset() for client in self.clients]
		
		# flatten list of lists and create a single dataset
		aggregated_data = [item for sublist in client_datasets for item in sublist]
		data_tensors = torch.cat([data for data, _ in aggregated_data])
		label_tensors = torch.cat([label for _, label in aggregated_data])
		
		# create a unified DataLoader from the aggregated data
		aggregated_dataset = TensorDataset(data_tensors, label_tensors)
		unified_loader = DataLoader(aggregated_dataset, batch_size=self.batch_size, shuffle=True)
		return unified_loader


	# central training simply means that each of the clients should send their data to the server
	# and the server should train the model on the data that it has recieved
	# clients: the clients that are going to be used in the training process
	def central_training(self):
		
		# collect all data from clients
		central_loader = self.aggregate_data_from_clients()

		# train centrally
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

		# set the model to training mode
		self.model.train()
  
		# train the model on the central data
		for data, target in central_loader:
			
			# move the data to the right device
			data, target = data.to(self.device), target.to(self.device)

			# reset the optimizer
			optimizer.zero_grad()

			# get the output of the model
			output = self.model(data)

			# compute the loss and backpropagate
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
    
    
	# define a function to train the model in the federated environment
	def federated_training(self):
  
		# keep track of the weights that have been returned from the clients
		client_weights = []
  
		# set everything up different depending on the environment that we are training in
		for i, client in enumerate(self.clients):
      
			# get the weights of the model
			weights = client.get_weights()
   
			# add the weights to the server
			client_weights.append(weights)
   
		# aggregate the weights
		self.client_weights = client_weights
		self.aggregate_parameters()
	  
		
  
  	
 
	# define a function to train the model in the distributed environment
	def distributed_training(self):
		
		# initialize random noise in the shape of the model gradients
		aggregated_gradients = {name: torch.zeros_like(param, device=self.device) 
								for name, param in self.model.named_parameters()}
  
		# add noise to the gradients
		for name, param in aggregated_gradients.items():
      
			# add noise to the gradients
			mean = torch.randn_like(param, device=self.device)
			std_dev = torch.randn_like(param, device=self.device)
			param += torch.randn_like(param, device=self.device) * std_dev + mean
   
		# set each client's next client to be the next client in the list
		# of course, this can be done randomly and the adversary can ruin the training
		# until they are one apart
		for i, client in enumerate(self.clients[:-1]):
			client.next_client = self.clients[(i + 1) % len(self.clients)]
   
		# set the last client's next client to be the server
		self.clients[-1].next_client = None
  
		# start the distributed training with the first client
		self.clients[0].distributed_train(aggregated_gradients, 0)
	
    
	# define the function that is going to be called at the end of distributed training
	# by the final client
	def distributed_training_end(self, total_gradients):
     
		# apply the aggregated gradients to the server model parameters
		for name, param in self.model.named_parameters():
			param -= total_gradients[name] / len(self.clients)
	
    
    
	# function to set up the distributed environment so that each of the clients in the model
	# has the same weights as the server model
	def reset_distributed_weights(self):

		# set everything up different depending on the environment that we are training in
		for i, client in enumerate(self.clients):
		
			# set the model weights to be the same as the server weights
			client.set_weights(self.model.state_dict())
		

