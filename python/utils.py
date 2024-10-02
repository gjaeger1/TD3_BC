import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	
def analyze_data(data: ReplayBuffer):
	# print dimenisons of the data
	print(f"State dim: {data.state.shape}")
	print(f"Action dim: {data.action.shape}")
	print(f"Next state dim: {data.next_state.shape}")
	print(f"Reward dim: {data.reward.shape}")
	print(f"Not done dim: {data.not_done.shape}")

	# print number of samples
	print(f"Number of samples: {data.size}")

	# print number of not done samples
	print(f"Number of not done samples: {np.sum(data.not_done)}")

	# print ratio of not done samples to overall number of samples
	print(f"Ratio of not done samples: {np.sum(data.not_done)/data.size}")


	# print min and max done signal
	print(f"Min done signal: {np.min(data.not_done)}")
	print(f"Max done signal: {np.max(data.not_done)}")

	# print min/max/mean/std of rewards
	print(f"Min reward: {np.min(data.reward)}")
	print(f"Max reward: {np.max(data.reward)}")
	print(f"Mean reward: {np.mean(data.reward)}")
	print(f"Std reward: {np.std(data.reward)}")

	# print min/max/mean/std of states for each dimension
	print(f"Min state: {np.min(data.state, axis=0)}")
	print(f"Max state: {np.max(data.state, axis=0)}")
	print(f"Mean state: {np.mean(data.state, axis=0)}")
	print(f"Std state: {np.std(data.state, axis=0)}")
	
	# print min/max difference between state and next state for each dimension
	print(f"Min state-next state difference: {np.min(data.state - data.next_state, axis=0)}")
	print(f"Max state-next state difference: {np.max(data.state - data.next_state, axis=0)}")


