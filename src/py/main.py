# DDPG
import torch
import gym 
from DDPGAgent import DDPGAgent
import numpy as np


env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim)

# hyperparameters
max_episodes = 200
max_steps = 200
batch_size = 64

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = agent.get_action(state) + np.random.normal(0, 0.1, action_dim)
        next_state, reward, done, _= env.step(2*action)
        agent.memory.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = next_state

        if done:
            break

    print('Episode: {}, Total reward: {}'.format(episode, episode_reward))

env.close()

#save model
torch.save(agent.actor.state_dict(), r'E:\VSpro\Py\RL_exp\Pendulum_sim_py\model\actor.pth')
torch.save(agent.critic.state_dict(), r'E:\VSpro\Py\RL_exp\Pendulum_sim_py\model\critic.pth')

