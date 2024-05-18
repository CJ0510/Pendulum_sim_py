import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DDPGAgent import Actor
import pygame


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(r'E:\VSpro\Py\RL_exp\Pendulum_sim_py\model\actor.pth'))

pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

episode = 5
max_steps = 200
for _ in range(episode):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        frame = env.render(mode='rgb_array')
        frame = np.transpose(frame, (1, 0, 2))
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (500, 500), screen)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        action = actor(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        state = next_state

        clock.tick(30)

        if done:
            break

    print('Episode: {}, Total reward: {}'.format(_, episode_reward))