import numpy as np
import torch
import argparse
import os
import csv

import utils
import TD3_BC

# local function to convert string to numpy array
def to_numpy_array(string):
	# Remove the brackets and newlines and split the string into individual numbers
	numbers = string.strip('[]').replace('\n','').split()
	return np.array(numbers, dtype=float)

def find_csv_files(directory):
	# find all csv files recursively and list their absolute path
	csvs = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith(".csv"):
				csvs.append(os.path.join(root, file))

	return csvs

def read_to_replay_buffer(filepath, replay_buffer = None, ignore_velocities = False):
	# Initialize variables to prevent empty files from causing errors
	max_action = 0
	state_dim = 0
	action_dim = 0

	# Load replay buffer from CSV file 
	with open(filepath, newline='') as csvfile:
		print(f"Reading {filepath}")
		reader = csv.DictReader(csvfile)
		max_action = 0
		for row in reader:
			obs = to_numpy_array(row['observation'])
			next_obs = to_numpy_array(row['next_observation'])
			reward = float(row['reward'])
			done = True if row['done'].lower() == 'true' else False
			action = to_numpy_array(row['action'])

			if ignore_velocities:
				# ignore last two elements of the observation
				obs = obs[:-2]
				next_obs = next_obs[:-2]

			max_action = max(max_action, np.max(np.abs(action)))

			state_dim = obs.shape[0]
			action_dim = action.shape[0]

			if done:
				next_obs = np.zeros_like(obs)
				action = np.zeros_like(action)

			if replay_buffer is None:
				print(state_dim, action_dim)
				replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
			
			if not obs.shape[0] == state_dim:
				print(obs)
				print(row['observation'])

			if not action.shape[0] == action_dim:
				print(action)
				print(row['action'])

			if not next_obs.shape[0] == state_dim:
				print(next_obs)
				print(row['next_observation'])

			replay_buffer.add(obs, action, next_obs, reward, done)

	return replay_buffer, state_dim, action_dim, max_action

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--replay_buffer", default="")          # Path to CSV file holding a dataset to be used as a replay buffer
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e10, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters default: False
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	# Data analysis
	parser.add_argument("--print_data_statistics", action="store_true") 
	parser.add_argument("--ignore-velocities", action="store_true")

	args = parser.parse_args()

	file_name = f"{args.policy}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	max_action = 0
	state_dim = 0
	action_dim = 0

	replay_buffer = None
	if args.replay_buffer:
		csvs = find_csv_files(args.replay_buffer)
		for csvfile in csvs:
			replay_buffer, state_dim, action_dim, max_action_tmp = read_to_replay_buffer(csvfile, replay_buffer)
			max_action = max(max_action_tmp, np.max(np.abs(max_action)))

	mean,std = replay_buffer.normalize_states()

	# ensure that state_dim and action_dim are set
	assert state_dim > 0
	assert action_dim > 0

	# print statistics of replay buffer
	if args.print_data_statistics:
		utils.analyze_data(replay_buffer)

	# preparation of policy
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			if args.save_model: policy.save(f"./models/{file_name}", mean=mean, std=std, use_torch_script=True)

	if args.save_model: policy.save(f"./models/{file_name}", mean=mean, std=std, use_torch_script=True)

	# print example outputs of model to check if it is working
	# Sample replay buffer 
	state, action, next_state, reward, not_done = replay_buffer.sample(args.batch_size)

	with torch.no_grad():
		# loop through states and select actions
		for i in range(state.shape[0]):
			# Select action according to policy and add clipped noise
			action = policy.select_action(state[i])
			print(f"State: {state[i]}")
			print(f"Action: {action}")
