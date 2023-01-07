import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D
import pygame

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -700 , high= 700 , shape=(8,), dtype=np.int64)

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.isRoundFinished()
        #print(self.pygame.alive)
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()

    def close(self):
        pygame.quit()

    def get_action_meanings(self):
        return []