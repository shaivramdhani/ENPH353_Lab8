#!/usr/bin/env python3
import random
import gym_gazebo 
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import random

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                saveQ(q_table, "best_q_table.npy")
                print("New highest reward: {total_reward}.")

            state,_= env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    #Get currrent Q value 
    current_q = q_table[state,action]

    #if robot loses line
    if next_state == -1:
        max_next_q = 0

    else:
        #Find the highest Q value available in the next step
        max_next_q = np.max(q_table[next_state])


    #calculate updated q value - uses the bellman algorithm or whatever
    new_q = current_q + alpha* (reward + gamma * max_next_q - current_q)

    #Update q vable
    q_table[state,action] = new_q


    return q_table


def choose_action(state, q_table, epsilon, action_space):
    action = 0

    #choose a random number between 0 and 1
    if random.random() < epsilon:
        #Explore by choosing a random action
        action = action_space.sample()

    else:
        #Exploititation

        #find highest q value
        max_q = np.max(q_table[state])

        #find all actions that have this q value
        best_actions = np.where(q_table[state] == max_q)[0]

        #Randomly choose between equal actions
        action = np.random.choice(best_actions)
    return int(action)


def saveQ(q_table, save_path):

    np.save(save_path,q_table)
    return

def loadQ(load_path):

    return np.load(load_path)

if __name__ == "__main__":
    main()