# this is the definition file for a single client in the federated learning environment
# which is going to recieve the model weights, compute the gradient with respect to the local update
# that it has, and then return the gradient before recieving the global gradient update
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import HealthNet

# the client class
# this is one of a series of clients that are going to be used in the federated learning process
class Client:

	# initialize the client with the dataset that it is going to be using and device that it should be running on
	# model: the model that the client is going to be using
	# dataset: the dataset that the client is going to be using
	# device: the device that the client is going to be running on
	# adversarial: whether the client is adversarial or not
	# adversarial_level: the level of adversarialness that the client has
	def __init__(self, 
					dataset: torch.utils.data.Subset, 
					device: torch.device,
					num_epochs: int = 5,
					learning_rate: float = 0.01,
					batch_size: int = 8,
					adversarial: bool = False,
					adversarial_level: int = 0,
					adversarial_method: str = 'none'
				):
		
		self.model = HealthNet(30).to(device)
  
		self.dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		self.device = device
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.adversarial = adversarial
		self.adversarial_level = adversarial_level
		self.adversarial_method = adversarial_method
  
		# the aggregated gradients over training
		self.agg_grads = None
		self.reset_agg_gradients()

		# this is only used when we are using distributed learning 
		self.next_client = None
  
		# if this is an adversarial client define a variable that defines whether this is the 
  		# first or second pass utilizing the data for a given training loop
		# we assume that there are not other clients that are adversarial or fail in the loop
		if self.adversarial:
			self.first_pass = True
			self.saved_gradients = None

  
	# function to set the weights of the model
	# weights: the weights that the model should be set to
	def set_weights(self, weights):
		self.model.load_state_dict(weights)
  
	# define a function to reset the aggregated gradients over training
	def reset_agg_gradients(self):
		self.agg_grads = {name: torch.zeros_like(param, device=self.device) 
							for name, param in self.model.named_parameters()}
  

	# function to train the model for some number of epochs and return the weights
	# epochs: the number of epochs that the model should be trained for
	# learning_rate: the learning rate that the model should be trained with
	def get_weights(self, epochs=-1, learning_rate=0.01):
     
		# check if we want to be training for some right number of epochs
		if epochs == -1:
			epochs = self.num_epochs
		
		# if the client is adversarial, then we should run the adversarial training
		if self.adversarial:
			self.adversarial_train(epochs, learning_rate)
		else:
			self.normal_train(epochs, learning_rate)
   
		# return the model weights
		return self.model.state_dict()


	# function to train the model for some number of iterations and return the gradients
	# epochs: the number of epochs that the model should be trained for
	# learning_rate: the learning rate that the model should be trained with	
	def get_gradients(self, epochs=1, learning_rate=0.01):
     
		# reset the gradients
		self.reset_agg_gradients()
     
		# if the client is adversarial, then we should run the adversarial training
		if self.adversarial:
			self.adversarial_train(epochs, learning_rate, save_gradients=True)
		else:
			self.normal_train(epochs, learning_rate, save_gradients=True)
   
		# return the model gradients
		return self.agg_grads


	# define a function that does adversarial training with the model
	def adversarial_train(self, epochs, learning_rate, save_gradients=False):
	
		# define the loss function and the optimizer for the model over this one training session
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

		# set the model to training mode
		self.model.train()
  
		# train the model for the number of epochs that are specified
		for _ in range(epochs):
	  
			# initialize a variable to store the total loss
			total_loss = 0
	  
			# iterate over the dataset
			for data, target in self.dataset:
		
				# move everything to the right device
				data, target = data.to(self.device), target.to(self.device)
		
				# zero the gradients
				optimizer.zero_grad()
		
				# get the prediction of the model
				output = self.model(data)
		
				# calculate the loss and backpropagate
				loss = criterion(output, target)

				# update model parameters
				loss.backward()

				# poison the gradients if we are supposed to
				if self.adversarial_method == 'poison':

					# invert the gradients
					with torch.no_grad():
						for param in self.model.parameters():
							if param.grad is not None:
								# invert the gradient
								param.grad *= -1 * 2 * self.adversarial_level

		
				# add noise to the gradients of the model
				elif self.adversarial_method == 'noise':
					
					for param in self.model.parameters():
						
						# only parameters with gradients need noise addition
						if param.grad is not None:  
							noise = torch.randn_like(param.grad) * self.adversarial_level
							param.grad += noise
					
				# update parameters using the noisy gradients
				optimizer.step()
		
				# save the gradients if we are supposed to
				if save_gradients:
					for name, param in self.model.named_parameters():
						if param.grad is not None:
							self.agg_grads[name] += param.grad.clone()
				
				# accumulate the loss
				total_loss += loss.item()
			
			# calculate the average loss for the epoch
			avg_loss = total_loss / len(self.dataset)
   
			# print the average loss
			# print(f"Epoch {_+1} Average Loss: {avg_loss}")

  
		

	# define a function that does normal training with the model
	# epochs: the number of epochs that the model should be trained for
	# learning_rate: the learning rate that the model should be trained with
	# save_gradients: whether we should save the gradients as we train
	def normal_train(self, epochs, learning_rate, save_gradients=False):
     
		# reset the gradients
		self.agg_grads = []

		# define the loss function and the optimizer for the model over this one training session
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

		# set the model to training mode
		self.model.train()
		
		# train the model for the number of epochs that are specified
		for e_num in range(epochs):
			
   			# initialize a variable to store the total loss
			total_loss = 0
			
			# iterate over the dataset
			for data, target in self.dataset:
				
				# move everything to the right device
				data, target = data.to(self.device), target.to(self.device)
				
				# zero the gradients
				optimizer.zero_grad()
				
				# get the prediction of the model
				output = self.model(data)
				
				# calculate the loss and backpropagate
				loss = criterion(output, target)
				loss.backward()
				
				# save the gradients if we are supposed to
				if save_gradients:
					self.agg_grads.append({name: param.grad for name, param in self.model.named_parameters()})
				
				# update model parameters
				optimizer.step()
				
				# accumulate the loss
				total_loss += loss.item()
	
			# print the loss
			# print(f"Epoch {e_num+1} Loss: {total_loss / len(self.dataset)}")
    
    
    # define the function that is going to be called if we are doing distributed learning
	def distributed_train(self, aggregated_gradients, num_in_sequence):
     
		# if this is the first pass, then we should send the data to the next client
		if self.first_pass:
		
  			# train the model
			self.normal_train(self.num_epochs, self.learning_rate, save_gradients=True)
	
			# add the gradients to the aggregated gradients
			for name, param in self.agg_grads.items():
				aggregated_gradients[name] += param.grad
    
			# now save a clone of the aggregated gradients in the local variable
			if self.adversarial:
				self.saved_gradients = {name: param.grad.clone() for name, param in self.agg_grads.items()}
		
		else:
			
			# we do not need to train the model again
			# we just need to pass the gradients to the next client
			# but we can also run the data attack on the delta in the gradients
   
			# compute the difference in the first saved gradients against these gradients
			delta_gradients = {name: param.grad - self.saved_gradients[name] for name, param in self.agg_grads.items()}
   
			# run the data attack on the delta gradients
			self.inference_attack(delta_gradients)
   
			# add the aggregated gradients from training to the aggregated gradients
			for name, param in self.agg_grads.items():
				aggregated_gradients[name] += param.grad


		if self.adversarial:
			
			# clear the gradients if this is the second pass
			if not self.first_pass:
				self.saved_gradients = None
      
			# invert the first pass parameter
			self.first_pass = not self.first_pass
    
		# call the distributed training funciton on the next client
		if self.next_client:
			self.next_client.distributed_train(aggregated_gradients, num_in_sequence + 1)
		else:
			self.distributed_training_end(aggregated_gradients)



	# define a function to run the inference attack
	def inference_attack(self, delta_gradients):
		raise NotImplementedError('Inference attack not implemented yet.')


	# define a function that returns the dataset that we are using for training this client
	def get_dataset(self):
		
		# return all the data from the dataloader
		return [(data, label) for data, label in self.dataset]

