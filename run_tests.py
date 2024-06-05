# this program is a python script that is meant to run a series of attacks on the federated learning framework we have implemented
# the program can be run with arguments to specify the environment and the attack to be run

# arguments to be passed to the program are as follows:
# --env: the environment to be used for the attack
# 		- options: 'central', 'federated', 'distributed'
# 			- 'central': run the training with a central server that does all updates using data aggregated from clients.
# 				       	This is the default environment and is the almost the same as a normal training process with one model.
# 			- 'federated': run the training with a federated learning setup that has clients update the server by passing the weights
#                       directly back to the server. The server then aggregates the weights and updates the model.
# 			- 'distributed': run the training with a distributed learning setup that has clients update the server by passing the gradients
#                       from client to client before sending the final gradients back to the server. The server then aggregates the gradients and updates the model.
#  			- 'all': run the attack on all environments in sequence

# --attack: the attack to be run on the environment
# 		- options: 'noise', 'inference', 'poison'
# 			- 'noise': some adversary adds noise to the gradients before sending them along
# 			- 'inference': some adversary infers the data that was used to update the model from other clients
# 			- 'poison': some adversary poisons the model by sending bad data along that is crafted to harm the total update
# 			- 'none': no attack is run

# --num_clients: the number of clients to be used in the training process
# 		- default: 5

# --num_rounds: the number of rounds to be run in the training process
# 		- default: 10

# --num_epochs: the number of epochs to be run in the training process
# 		- default: 1

# --learning_rate: the learning rate to be used in the training process
# 		- default: 0.01

# --attack_intensity: the intensity of the attack to be run. It is on a scale from 1 to 5 with 5 being the most intense attack. 
# 					This correlates with the level of noise in the noise attack, the magnitude of the poisoning in the poison attack, etc. 
# 					Although the values are pre-defined by the author, trivial manipulation of the code can allow for any value to be used.
# 					This does not affect the inference attack
# 		- default: 3

# --verbose: the level of verbosity that the program should run at
# 		- options: '0', '1', '2'



# import the necessary libraries
import torch
import random
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split

from Base_Code import Client, Server, HealthNet

# the batch size that each of the loaders will use
BATCH_SIZE = 8


# define a function that parses the arguments that are passed to the program
def parse_arguments():
	
	# create the parser
	parser = argparse.ArgumentParser(description="Run federated learning experiments with attacks.")
	
	# get the arguments
	parser.add_argument('--env', type=str, default='central', choices=['central', 'federated', 'distributed', 'all'],
						help='The environment to use for the attack.')
	parser.add_argument('--attack', type=str, default='none', choices=['noise', 'inference', 'poison', 'none'],
						help='The type of attack to run.')
	parser.add_argument('--num_clients', type=int, default=10, help='Number of clients in the federated learning setup.')
	parser.add_argument('--num_rounds', type=int, default=10, help='Number of training rounds.')
	parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs each client trains per round.')
	parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training.')
	parser.add_argument('--attack_intensity', type=int, default=3, choices=[1, 2, 3, 4, 5], help='Intensity of the attack, scale 1-5.')
	parser.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2],
						help='The level of verbosity that the program should run at.')
	
	# return the values
	return parser.parse_args()




# this function gets the data from the file that we want it to
# it is currently hard coded to use the data.csv dataset, but this can be 
# changed through changing the model structrue and changing the 
# code for loading below
# filepath: the path to the file that we want to load
# num_clients: the number of clients that we want to split the data between
# validation_split: the percentage of the data that should be used for validation
def load_and_preprocess_data(filepath, num_clients, validation_split=0.1):
    
	# get the data
	data = pd.read_csv(filepath)

	# drop the id and other columns that make it so the model doesn't generalize
	data = data.drop(columns=['id', 'Unnamed: 32'])

	# encode the benign and malignant labels
	label_encoder = LabelEncoder()
	data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

	# extract features and labels
	features = data.drop('diagnosis', axis=1)
	labels = data['diagnosis']

	# normalize the featuress
	scaler = StandardScaler()
	features = scaler.fit_transform(features)

	# change them so that they are tensors and reshape them for the loss
	features_tensor = torch.tensor(features, dtype=torch.float32)
	labels_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1) 

	# return the dataset
	dataset = TensorDataset(features_tensor, labels_tensor)

	# split the dataset equally among the clients
	total_size = len(dataset)
	val_size = int(total_size * validation_split)
	train_size = total_size - val_size
 
	print(f"Length of dataset: {total_size}")
	print(f"Length of training dataset: {train_size}")
	print(f"Length of validation dataset: {val_size}")

	# split the dataset into train and validate
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	# get the sizes of the data for each client
	sizes = [train_size // num_clients] * num_clients

	# handle the remainder of the data
	sizes[-1] += train_size - sum(sizes)
 
	# get the data subsets
	train_subsets = random_split(train_dataset, sizes)

	return train_subsets, val_dataset



# # function to run the inference attack
# def run_inference_attack():
    
# 	# choose a random index that is not the last index or the index before the last index
# 	index = random.randint(0, num_clients - 3)

# 	# assign dual adversarial clients
# 	clients[index] = Client(dataset=client_datasets[i], device=device, num_epochs=num_epochs, learning_rate=learning_rate, adversarial=True, adversarial_level=attack_intensity, adversarial_method='inference')
# 	clients[index + 2] = [clients[index]]


# define a function that takes the server model and runs it on the validation dataset to get the 
# accuracy and loss of the model
def validate_model(model, dataset, device):
    
	# set the model to evaluation mode
	model.eval()

	# get the data loader
	loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	# initialize the total loss and accuracy count
	total_loss = 0
	accuracy_count = 0

	# define the loss function
	criterion = torch.nn.MSELoss()

	# loop through the data
	for data, target in loader:
		# move the data to the right device
		data, target = data.to(device), target.to(device)
		
		# get the output of the model
		output = model(data)
		
		# compute the loss
		total_loss += criterion(output, target).item()
		
		# apply thresholds to output for custom accuracy calculation
		output = output.squeeze()  # Ensure output is of correct shape, adjust as necessary
		predictions = (output > 0.5).float() * (target > 0.5).float() + (output < 0.5).float() * (target < 0.5).float()
		
		# count accurate predictions
		accuracy_count += predictions.sum().item()

	# calculate the average loss and accuracy over all batches
	average_loss = total_loss / len(loader)
	accuracy = accuracy_count / (len(dataset) * BATCH_SIZE)

	# return the average loss and accuracy
	return average_loss, accuracy



# the main function that is going to be run
def main(data_file='./data.csv'):
	
	# get the arguments
	args = parse_arguments()

	env = args.env
	attack = args.attack
	num_clients = args.num_clients
	num_rounds = args.num_rounds
	num_epochs = args.num_epochs
	learning_rate = args.learning_rate
	attack_intensity = args.attack_intensity
	verbosity_level = args.verbose
 
	# get the device that we should be training on
	if torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
	
	fed_attack_ascii = '''
 _______________________________________________________________
 ______ ______ _____          _______ _______       _____ _  __
 |  ____|  ____|  __ \      /\|__   __|__   __|/\   / ____| |/ /
 | |__  | |__  | |  | |    /  \  | |     | |  /  \ | |    | ' / 
 |  __| |  __| | |  | |   / /\ \ | |     | | / /\ \| |    |  <  
 | |    | |____| |__| |  / ____ \| |     | |/ ____ \ |____| . \ 
 |_|    |______|_____/  /_/    \_\_|     |_/_/    \_\_____|_|\_\\
 _______________________________________________________________
                                                                '''
                                                                
	print(fed_attack_ascii)
  
	# print the arguments
	print("Arguments:")
	print(f"Environment: {env}")
	print(f"Attack: {attack}")
	print(f"Number of Clients: {num_clients}")
	print(f"Number of Rounds: {num_rounds}")
	print(f"Number of Epochs: {num_epochs}")
	print(f"Learning Rate: {learning_rate}")
	print(f"Attack Intensity: {attack_intensity}")
	print(f"Verbose: {verbosity_level}")

	# print the device
	print(f"Device: {device}")
	print()
  
 
	# load the data
	client_datasets, val_dataset = load_and_preprocess_data(data_file, num_clients)
 
	# client list for the server
	clients = []
 
	# create the clients
	for i in range(num_clients):
		client = Client(dataset=client_datasets[i], device=device, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=BATCH_SIZE, adversarial=False, adversarial_level=0, adversarial_method='none')
		clients.append(client)
  
  
	# if the attack method is inference, then two of the clients are going to be working together and adversarial
	# below we are going to replace the clients with the adversarial clients
	# the clients should have at least one client between them that is not adversarial
	# there are lots of ways to get them exactly one client apart depending on the implementation, so we assume that they are one apart
	if attack == 'inference':
     
		# choose a random index that is not the last index or the index before the last index
		index = random.randint(0, num_clients - 3)

		# assign dual adversarial clients
		clients[index] = Client(dataset=client_datasets[i], device=device, num_epochs=num_epochs, learning_rate=learning_rate, adversarial=True, adversarial_level=attack_intensity, adversarial_method='inference')
		clients[index + 2] = [clients[index]]
  
	elif attack == 'noise' or attack == 'poison':
     
		# choose a random client to be adversarial
		index = random.randint(0, num_clients - 1)
		clients[index] = Client(dataset=client_datasets[i], device=device, num_epochs=num_epochs, learning_rate=learning_rate, adversarial=True, adversarial_level=attack_intensity, adversarial_method=attack)
  
	elif attack == 'all':
		raise NotImplementedError('All attacks are not implemented yet.')

  
	# create the server and the clients
	server = Server(device=device, clients=clients, environment=env, num_rounds=num_rounds, batch_size=BATCH_SIZE)
 
	# run the training rounds
	server.setup_distributed_environment()
 
	# start the training
	server.run(num_rounds=num_rounds)

	# validate the model and get the statistics
	loss, accuracy = validate_model(server.model, val_dataset, device)

	# print the state dictionary 
	if verbosity_level > 1:
		print(f"State Dictionary: {server.model.state_dict()}")

	# print the statistics
	print(f"RESULTS")
	print(f"Validation Loss: {loss:.4f}")
	print(f"Validation Accuracy: {(accuracy * 100):2f}%")


if __name__ == "__main__":
    main()