# this is the program that should be used to run the entire federated learning training process
# the program is currently un-optimized in that it does not train the models simultaneously
# and further development of this project should all for this to happen

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from client import Client
from server import Server


# this function gets the data from the file that we want it to
# it is currently hard coded to use the data.csv dataset, but this can be 
# changed through changing the model structrue and changing the 
# code for loading below
def load_and_preprocess_data(filepath):
    
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
    return TensorDataset(features_tensor, labels_tensor)

# this function is used to split the gradients between
# each of the clients that are being used in this training process
def distribute_data_to_clients(dataset, num_clients):

	# split the dataset
	total_size = len(dataset)
	each_size = total_size // num_clients

	# get the data subsets
	subsets = random_split(dataset, [each_size]*num_clients + [total_size - each_size*num_clients])[:num_clients]

	return subsets

# config for program
filepath = '../data.csv'
num_clients = 5

# load the data
dataset = load_and_preprocess_data(filepath)

# get the data chunks
client_datasets = distribute_data_to_clients(dataset, num_clients)

# create the server and the clients
server = Server()
clients = [Client(client_datasets[i]) for i in range(num_clients)]

# run the training rounds
num_rounds = 10
for round in range(num_rounds):
    
    print(f"Round {round+1} starts")
 
    # client update from training
    client_updates = [client.get_gradients() for client in clients]
    
    # server aggregates updates
    server.federated_avg(client_updates)
    
    # distribute the new model to each of the clients
    global_model_weights = server.get_global_model().state_dict()
    for client in clients:
        client.set_weights(global_model_weights)

	# logging
    print(f"Round {round+1} completed")
