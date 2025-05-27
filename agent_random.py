"""
creates an random agent in the ALE
- serves as a basic baseline
- taking actions uniformly at random within the environment without any learning or strategic decision-making
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import NoopResetEnv, EpisodicLifeEnv, WarpFrame, FireResetEnv, ClipRewardEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import wandb
import time
import numpy as np
import random
from typing import List, Dict
from ale_py import ALEInterface

# --- Custom Wrapper from your block learning code ---
class GameNameWrapper(gym.Wrapper):
    """Adds game_name to info dict during evaluation."""
    def __init__(self, env, game_name: str):
        super().__init__(env)
        self.game_name = game_name

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info['game_name'] = self.game_name
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info['game_name'] = self.game_name
        return obs, reward, done, truncated, info

def make_eval_env(game_name: str):
    """
    Creates an evaluation environment for a given Atari game with the same wrappers.

    Args:
        game_name (str): Name of the Atari game (e.g., "SpaceInvaders").

    Returns:
        the environemnt with all the wrappers applied 
    """
    env = gym.make(f'ALE/{game_name}-v5', full_action_space=True)
    env = Monitor(env)  # Enables logging of episode statistics
    env = NoopResetEnv(env, noop_max=30)
    env = WarpFrame(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = GameNameWrapper(env, game_name)
    return env

def evaluate_random_agent(game_name: str, n_eval_episodes: int = 5):
    """
    Evaluates a random agent on the specified game.
    
    Args:
        game_name (str): Name of the Atari game (e.g., "SpaceInvaders").
        n_eval_episodes (int): Number of episodes to run for evaluation.
        
    Returns:
        mean_reward, std_reward, mean_length
    """
    env = make_eval_env(game_name)
    all_rewards = []
    all_lengths = []
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        while True:
            action = env.action_space.sample()  # Random action!
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if done or truncated:
                break
        print(f"[{game_name}] Episode {episode+1}: reward = {ep_reward}, length = {ep_length}")
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
    
    env.close()
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)
    
    print(f"\nRandom Agent Baseline on {game_name}:")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Std Reward: {std_reward:.2f}")
    print(f"  Mean Episode Length: {mean_length:.2f}\n")
    
    return mean_reward, std_reward, mean_length


def simulate_random_agent_run(total_timesteps: int = 30_000_000,
                              eval_freq: int = 30_000,
                              n_eval_episodes: int = 5,
                              game_name: str = "SpaceInvaders"):
    """
    Simulate a training run for a random agent by evaluating its performance every `eval_freq` timesteps.
    The random agent does not learn, so its performance remains constant, but this will give you a timeline 
    that you can compare with your PPO runs.
    """
    # Initialize WandB (make sure to call wandb.finish() later if needed)
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
        notes= config['notes']
    )

    # Simulate the timeline by logging evaluations at the same intervals as your PPO agent
    current_timesteps = 0
    while current_timesteps < total_timesteps:
        print(f"Evaluating at timestep: {current_timesteps}")
        mean_reward, std_reward, mean_length = evaluate_random_agent(game_name, n_eval_episodes)
        
        game_name = config["game"]
        wandb.log({
            f"eval/{game_name}_mean_reward": mean_reward,
            f"eval/{game_name}_std_reward": std_reward,
            f"eval/{game_name}_mean_length": mean_length,
            "timesteps": current_timesteps
        }, step=current_timesteps)

        current_timesteps += eval_freq

    wandb.finish()
    print("Random agent evaluation complete.")


games = ['Phoenix', 'Galaxian', "SpaceInvaders"]
for i in range(3):
    if __name__ == "__main__":

        curr_game = games[i]

        config = {
            "project_name": "RandomAgent",
            "run_name": f"{curr_game}_20Mio",
            "notes": "RandomAgtent eval. 20Mio timesteps, eval every 30000",
            "game": f"{curr_game}",
            "timesteps":20_000_000,
            "n_envs": 8,
            "eval_freq": 30000,
            "n_eval_episodes": 5 #!!!!
        }

        # For example, simulate a random agent run on "SpaceInvaders" over 30M timesteps with evaluations every 20k timesteps.
        simulate_random_agent_run(total_timesteps=config["timesteps"],
                                eval_freq=config["eval_freq"],
                                n_eval_episodes=config["n_eval_episodes"],
                                game_name=config["game"])
