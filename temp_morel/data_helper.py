import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from typing import *



def load_data(data_path):
    if os.path.exists(data_path):
        data = np.load(data_path)
        return data
    else:
        raise FileExistsError("{} not exists".format(data_path))
    

class OfflineRLDataset(Dataset):

    def __init__(self, data_path, device):
        self.data = load_data(data_path=data_path)
        self.device = device
        self.normalize()

        self.total_size = self.data['obs'].shape[0]


    def __len__(self):
      return np.shape(self.data['obs'])[0]

    def __getitem__(self, idx):

        device = self.device

        obs = torch.from_numpy(self.data['obs'][idx]).to(device)
        action = torch.from_numpy(self.data['action'][idx]).to(device)
        next_obs = torch.from_numpy(self.data['next_obs'][idx]).to(device)
        reward = torch.from_numpy(self.data['reward'][idx]).to(device)
        done = torch.from_numpy(self.data['done'][idx]).to(device)
        return {
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "done": done
        }
    
    def get_sample_size(self):
        return self.data["obs"].shape[0]
    
    def get_obs_dim(self):
        return self.data["obs"].shape[1]
    
    def get_action_dim(self):
        return self.data["action"].shape[1]
    
    def normalize(self):
        dataset = self.data
        
        # Input data
        self.source_observation = dataset["obs"]
        self.source_action = dataset["action"]


        # Output data
        self.target_delta = dataset["next_obs"]- self.source_observation
        self.target_reward = dataset["reward"]

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean)/self.action_std
        self.source_observation = (self.source_observation - self.observation_mean)/self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean)/self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean)/self.reward_std

        # Get indices of initial states
        self.done_indices = dataset["done"] 
        self.done_indices = np.where(self.done_indices == "True", 1, 0)
        print(self.done_indices)
        self.initial_indices = np.roll(self.done_indices, 1)
        print(self.initial_indices)
        #self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)


    def get_reward_boundary(self) -> Tuple[float, float]:
        min_reward = self.data['reward'].min()
        max_reward = self.data['reward'].max()
        return min_reward, max_reward

    def get_value_boundary(self, gamma : float, enlarge_ratio : float = 0.2) -> Tuple[float, float]:
        min_reward, max_reward = self.get_reward_boundary()
        min_value = (min_reward - enlarge_ratio * (max_reward - min_reward)) / (1 - gamma)
        max_value = (max_reward + enlarge_ratio * (max_reward - min_reward)) / (1 - gamma)
        return min_value, max_value

    def get_action_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.action_space is None:
            return self.data['action'].min(axis=0), self.data['action'].max(axis=0)
        else:
            return self.action_space.low, self.action_space.high

    def get_obs_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.obs_space is None:
            return self.data['obs'].min(axis=0), self.data['obs'].max(axis=0)
        else:
            return self.obs_space.low, self.obs_space.high
        
    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        indexes = np.random.randint(0, self.total_size, size=(batch_size))
        return {k : v[indexes] for k, v in self.data.items()}

        


