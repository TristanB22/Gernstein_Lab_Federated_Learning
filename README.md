# Gernstein_Lab_Federated_Learning


### Purpose

This respository contains all of the code for creating a federated learning environment. The purpose of the project is to test different attacks on the example environments to expose where federated learning is insecure and can be improved. 


### Running the Project
In order to run the baseline federated learning process found in this repository, please navigate run the `run_tests.py` file. All code that is used in the run script can be found in the ```Base Code``` directory. 

The script accepts several options to tailor the federated learning environment and the type of attacks:

- **--env**: Specifies the learning environment to be used for the simulation.
  - `central`: A central server updates using aggregated client data. This is the default and simulates a standard centralized learning process.
  - `federated`: Clients update the server by directly passing their model weights, which the server aggregates and uses to update the global model.
  - `distributed`: Involves a chain of clients where each passes gradients to the next client before the final set of gradients is sent to the server for aggregation and model updating.
  - `all`: Executes simulations across all the specified environments in sequence.

- **--attack**: Determines the type of adversarial attack to apply during the training process.
  - `noise`: Adds noise to the gradients, simulating a scenario where data transmission or computation introduces errors.
  - `inference`: Simulates an attack where an adversary attempts to infer sensitive data from the model updates shared by other clients.
  - `poison`: Introduces deliberately harmful updates to corrupt the model and degrade its performance.
  - `none`: Runs the training without any adversarial interference.

- **--num_clients**: Sets the number of clients participating in the federated learning process.
  - Default: `5`

- **--num_rounds**: Specifies the number of training rounds to complete during the simulation.
  - Default: `10`

- **--num_epochs**: Defines how many epochs each client will train on their local data per round.
  - Default: `1`

- **--learning_rate**: Sets the learning rate for the optimization algorithm used in training.
  - Default: `0.01`

- **--attack_intensity**: Adjusts the intensity of the specified attack, on a scale from `1` (least intense) to `5` (most intense), influencing the magnitude or frequency of the attack's effect.
  - Default: `3`
  - This parameter does not affect the 'inference' attack type.

- **--verbose**: Adjusts the level of verbosity that the program should run at to help with debugging and development.
	- Type: `int`
	- Default: `0`

### Example Usage

To run the project in a federated environment with a noise attack, using five clients for ten training rounds, execute the following command:

```bash 
python3 run_tests.py --env federated --attack none --num_clients 6 --num_rounds 10 --num_epochs 3 --learning_rate 0.001 --attack_intensity 4 --verbose 0
```

This command configures the system to simulate a realistic scenario where client-side model training might be subjected to noise, affecting the quality of updates sent to the server.

This repository will be updated with more experiments that can be run as they are devised and created. 

### Data

The data that is used in this example project is provided by the [Dr. Wolberg at the University of Wisconsin Hospitals, Madison](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). The data is not included in the Github repository for the sake of space. However, the code can be trivially altered to allow for other datasets to be used instead. 

This project is supervised by the Gernstein Lab at Yale. Please reach out to the author, [Tristan Brigham](mailto:tristan.brigham@yale.edu) with any questions. 
