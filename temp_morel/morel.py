import os

import gym

from dynamic_models import *
from fake_env import FakeEnv
from policy.ppo2 import PPO2

from waterwork_env import * 


class Morel():
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dynamics_models = DynamicsEnsemble(obs_dim+action_dim, obs_dim+1+1, threshold=1.0)
        self.policy = PPO2(obs_dim, action_dim)

    def train(self, dataloader, dynamics_data):
        """
        Args:
            dataloader: Data loader providing batches of training samples 
                        (states, actions, rewards, next_states, dones) for the agent
            dynamics_data: Information about the environment dynamics, 
                        used for normalizing observations and actions,
                        serves as input to train the dynamic_model
        """
        self.dynamics_data = dynamics_data

        print("Beginning Training of Dynamic Models -----")
        ## TODO: training a ensemble of dynamic models 
        ##       and save them in the form of pretrained_model_x.pt
        ##  self.dynamics_models.train(dataloader, epoch)
        self.dynamics_models.train(dataloader, epochs=10)
        self.dynamics_models.save()
        del self.dynamics_models
        print("End Training of Dynamic Models ------")
        
        print("Begin Constructing PMDP ----")
        ## TODO: construct a Pessimistive MDP based on the 
        ##       pretrained dynamic models, using uncertainty 
        ##       estimation
        ##  env = fakeEnv(..)
        self.dynamics_models = DynamicsEnsemble(self.obs_dim+self.action_dim, self.obs_dim+1+1, threshold=1.0)
        self.dynamics_models.load()
        print(self.dynamics_models.models)
        env = FakeEnv(self.dynamics_models,
                    self.dynamics_data.observation_mean,
                    self.dynamics_data.observation_std,
                    self.dynamics_data.action_mean,
                    self.dynamics_data.action_std,
                    self.dynamics_data.delta_mean,
                    self.dynamics_data.delta_std,
                    self.dynamics_data.reward_mean,
                    self.dynamics_data.reward_std,
                    self.dynamics_data.initial_obs_mean,
                    self.dynamics_data.initial_obs_std,
                    self.dynamics_data.source_observation,
                    uncertain_penalty=-50.0)

        print("End Constructin PMDP ----")

        print("Begining Policy Training ----")
        ## TODO: Train a good online planning policy
        ## self.policy.train(env)
        self.policy.train(env)
        self.policy.save()
        del self.policy

        print("End Plicy Training ----")

        print("-----------Finish Training-------------------------")

    def eval(self):
        env = Waterworks()

        total_rewards = []
        for i in tqdm(range(100)):
            _, _, _, _, _, _, _, info = self.policy.generate_experience(env, 1024, 0.95, 0.99)
            total_rewards.extend(info["episode_rewards"])


        print("Final evaluation reward: {}".format(sum(total_rewards)/len(total_rewards)))


    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        self.policy.save(save_dir)  ## save the parameters of policy network
        self.dynamics_models.save(save_dir) ## save the parameters of dynamics network 
        ##? Should I save all the 10 dynamic_models together or implement a func
        ## to save them one by one

        print("saved policy network, dymamic models to {}".format(save_dir))


    def load(self, load_dir):
        if os.path.exists(load_dir):
            self.policy.load(load_dir)
            self.dynamics_models.load(load_dir)
        else:
            print("dir not exists")