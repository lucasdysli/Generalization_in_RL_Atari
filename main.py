"""
This file creates an agent and trains him either on pure pixels based (pixles) or with object-centric input (oc)

Training is done with different training stretegies:
1. baseline
2. normal_curriculum
3. random_curriculum

Training is always done with an PPO agent and fixed hyperparameters
"""

from ocatari_wrappers import BinaryMaskWrapper, ObjectTypeMaskWrapper, ObjectTypeMaskPlanesWrapper, MaskedBaseWrapper
from ocatari.core import OCAtari 


from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from typing import List, Dict
import gymnasium as gym
import time
import warnings
import random
import torch
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.atari_wrappers import NoopResetEnv, EpisodicLifeEnv, WarpFrame, FireResetEnv, ClipRewardEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

from ale_py import ALEInterface
import numpy as np

local_steps = {
    "Phoenix": 0,
    "SpaceInvaders": 0,
    "Galaxian": 0,
}

class ObjectTypeMaskPlanesWrapperLucas(MaskedBaseWrapper): 
    """
    A Wrapper that outputs a binary mask including
    only white bounding boxes of all objects on a black background, where
    every conceptually similar object type across specified games is on
    its own consistent plane, regardless of the object's name in a specific game.

    Designed for cluster generalization across SpaceInvaders, Galaxian, Phoenix.
    """

    def __init__(self, env: gym.Env, *args, **kwargs):
        # Define the unified mapping from object category name to plane index
        self.object_types = {
            # --- Conceptual Plane 0: Player ---
            "Player": 0,

            # --- Conceptual Plane 1: Player Projectile ---
            "Bullet": 1,             # Space Invaders
            "PlayerMissile": 1,      # Galaxian
            "Player_Projectile": 1,  # Phoenix

            # --- Conceptual Plane 2: Enemy / Alien / Target ---
            "Alien": 2,              # Space Invaders / Galaxian (script) / Phoenix (script)
            "EnemyShip": 2,          # Galaxian
            "DivingEnemy": 2,        # Galaxian (script) - Treat as enemy type
            "Phoenix": 2,            # Phoenix (enemy type)
            "Bat": 2,                # Phoenix (enemy type)
            "Boss": 2,               # Phoenix (main enemy target)
            "Satellite": 2,          # Space Invaders (often a target or bonus)

            # --- Conceptual Plane 3: Enemy Projectile ---
            "EnemyMissile": 3,       # Galaxian
            "Enemy_Projectile": 3,   # Phoenix (script)

            # --- Conceptual Plane 4: Shield / Obstacle ---
            "Shield": 4,             # Space Invaders / Galaxian (script) / Phoenix (script)
            "Boss_Block_Green": 4,   # Phoenix (obstacle part of boss)
            "Boss_Block_Blue": 4,    # Phoenix (obstacle part of boss)
            "Boss_Block_Red": 4,     # Phoenix (obstacle part of boss)
        }

        # Calculate the number of planes needed based on the highest index used + 1
        if not self.object_types:
             num_planes_needed = 0
        else:
             num_planes_needed = max(self.object_types.values()) + 1

        # Initialize the base wrapper with the calculated number of planes
        super().__init__(env, num_planes=num_planes_needed, *args, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """
        Sets the pixel values for a detected object 'o' on the correct plane.
        Uses the manually defined self.object_types mapping.
        """
        try:
            # Find the designated plane for this object's category
            plane_index = self.object_types[o.category]
            # Fill the object's bounding box with white (255) on that plane
            self.state[plane_index, y_min:y_max, x_min:x_max].fill(255)
        except KeyError:
            # Handle cases where an object category is detected but not in our map
            print(f"Warning: Object category '{o.category}' not found in object_types mapping. Skipping object.")
            pass # Skip this object if its category is unknown

class OCAtariToSB3Wrapper(gym.ObservationWrapper):
    """
    Class to convert the observation shape of the input, to make it compatible with OCAtari

    Convets the input from (C, H, W) -> (H, W, C)

    Returns:
        The converted input representation
    """
    def __init__(self, env):
        super().__init__(env)
        original_shape = self.env.observation_space.shape
        # Convert from (C, H, W) to (H, W, C)
        new_shape = (original_shape[1], original_shape[2], original_shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convert to uint8 and transpose axes
        return obs.astype(np.uint8).transpose(1, 2, 0)  # C,H,W -> H,W,C

class GameNameWrapper(gym.Wrapper):
    """
    This class adds game_name to info dict during evaluation. 
    This is necessary for future evaluation purposes.
    """
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

def lr_schedule(iterations):
    """
    returns a list with quartiles of iteartions
    e.g. iterations = 5
    -> [1, 0.8, 0.6, 0.4, 0.2, 0.0]
    or iterations = 3
    -> [1, 0.67, 0.33, 0.0]
    """ 
    lisa = [1] 
    rate = 1/iterations
    curr = 1
    def rekursion(lisa, rate, curr):
        if(round(curr,2) == 0.0):
            return
        val = curr - rate
        lisa.append(round(val,2))
        rekursion(lisa, rate, val)

    rekursion(lisa, rate, curr)
    return lisa

def linear_schedule_for_lr(initial_value, end_value):
        """
        Returns a function that computes a linearly decaying schedule
        from 'initial_value' down to end_value, as progress goes from 1 to 0.
        """
        def func(progress_remaining):
            return ( end_value + (initial_value - end_value) * progress_remaining ) 
        return func

def time_game_elapsed(start_time=None, game=None):
        '''
        Prints the elapsed game time of the training process

        Returns:
            the game ned time, and the corresponding hours, miutes and seconds
        '''
        if start_time is None:
            return 0, 0, 0, 0
        game_end_time = time.time()
        game_elapsed_time = game_end_time - start_time
        game_hours = int(game_elapsed_time // 3600)
        game_minutes = int((game_elapsed_time % 3600) // 60)
        game_seconds = int(game_elapsed_time % 60)
        print("Training on " + game + f" took {game_hours} hours, {game_minutes} minutes, and {game_seconds} seconds.")
        return game_end_time, game_hours, game_minutes, game_seconds

def make_evaluation_envs(game_names: List[str], base_seed: int) -> SubprocVecEnv:
    """
    Creates a parallelized evaluation environment for each game: Space invaders, Phoenix and Galaxian

    Args:
        game_names (List[str]): Names of the games to evaluate.
        base_seed (int): Seed for reproducibility.

    Returns:
        SubprocVecEnv: Vectorized environment running all specified games in parallel.
    """
    def make_env(game_name: str, seed: int):
        """
        Factory function to initialize an individual evaluation environment with consistent preprocessing and wrappers.
        """
        def _init():
            env = OCAtari(f'ALE/{game_name}-v5', obs_mode=config["ocatari_obs_mode"], 
                mode=config["ocatari_mode"], buffer_window_size= config["ocatari_buffer_size"], full_action_space=True)
            env = Monitor(env)
            env = NoopResetEnv(env, noop_max=30)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = GameNameWrapper(env, game_name)
            if config["oc_wrapper"] == "binary":
                env = BinaryMaskWrapper(env, include_pixels=config["ocwrapper_pixels"], 
                        buffer_window_size=config["ocwrapper_buffer_size"], work_in_output_shape=config["ocwrapper_work_in_output_shape"])
            elif config["oc_wrapper"] == "planes":
                env = ObjectTypeMaskPlanesWrapperLucas(env, buffer_window_size=config["ocwrapper_buffer_size"], 
                        include_pixels=config["ocwrapper_pixels"], work_in_output_shape=config["ocwrapper_work_in_output_shape"])
            elif config["oc_wrapper"] == "object":
                env = ObjectTypeMaskWrapper(env, buffer_window_size=config["ocwrapper_buffer_size"], include_pixels=config["ocwrapper_pixels"], 
                        work_in_output_shape=config["ocwrapper_work_in_output_shape"], v2=config["oc_object_v2"])
            env = OCAtariToSB3Wrapper(env)
            # Set seeds for evaluation env
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env
        return _init

    # Generate unique seeds for each evaluation game
    seeds = [base_seed + 1000 + i for i in range(len(game_names))]  # Offset to avoid overlap
    env_fns = [make_env(name, seed) for name, seed in zip(game_names, seeds)]
    return SubprocVecEnv(env_fns)

def env_custom_wrapper(game_name, seed=None):
    """
    Creates a training environment for a single game with all training-specific preprocessing steps.
    Includes rspecific object-centric wrappers.

    Args:
        game_name(str): Name of the game to train on
        seed (int): the current seed for reproducibility

    Rerurns:
        The training envornment with all wrappers applied
    """
    def make_env():
        env = OCAtari(f'ALE/{game_name}-v5', obs_mode=config["ocatari_obs_mode"], 
                mode=config["ocatari_mode"], buffer_window_size= config["ocatari_buffer_size"], full_action_space=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        if config["oc_wrapper"] == "binary":
            env = BinaryMaskWrapper(env, include_pixels=config["ocwrapper_pixels"], 
                buffer_window_size=config["ocwrapper_buffer_size"], work_in_output_shape=config["ocwrapper_work_in_output_shape"])
        elif config["oc_wrapper"] == "planes":
            env = ObjectTypeMaskPlanesWrapperLucas(env, buffer_window_size=config["ocwrapper_buffer_size"], 
                    include_pixels=config["ocwrapper_pixels"], work_in_output_shape=config["ocwrapper_work_in_output_shape"])
        elif config["oc_wrapper"] == "object":
            env = ObjectTypeMaskWrapper(env, buffer_window_size=config["ocwrapper_buffer_size"], include_pixels=config["ocwrapper_pixels"], 
                    work_in_output_shape=config["ocwrapper_work_in_output_shape"], v2=config["oc_object_v2"])
        env = OCAtariToSB3Wrapper(env)
        env = GameNameWrapper(env, game_name)

        # Set seed after all wrappers are applied
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return make_env


def extract_model_dir(games_list_train, global_i):
    """
    extract the corresponmding model directory for evalution procedure 
    """
    last_game = games_list_train[-1]
    return f"./models/normal_curriculum/ppo_glo{global_i}_l3{last_game}_{config['project_name']}_{config['run_name']}.zip"


"""
General Stuff above
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
normal_curriulum specific below
"""


class MultiEvalCallback(BaseCallback):
    """
    Callback for periodic parallel evaluation on all specified games.
    Logs performance metrics to Weights & Biases (wandb).
    """
    def __init__(self, eval_env: VecFrameStack, eval_game_names: List[str], 
                 eval_freq: int, n_eval_episodes: int = 5, global_offset: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_game_names = eval_game_names
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        self.global_offset = global_offset

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            game_stats = self._evaluate_parallel()

            global_step = self.global_offset + self.num_timesteps

            for game, stats in game_stats.items():
                wandb.log({
                    f"eval/{game}_mean_reward": stats['mean_reward'],
                    f"eval/{game}_std": stats['std_reward'],
                    f"eval/{game}_mean_ep_length": stats['mean_length'],
                    "timesteps": global_step
                }, step=global_step)
                if self.verbose > 0:
                    print(f"Eval {game}: mean_reward={stats['mean_reward']:.2f}")
        return True

    def _evaluate_parallel(self) -> Dict[str, Dict]:
        """Evaluate on all games in parallel."""
        episodes_counts = {game: 0 for game in self.eval_game_names}

        all_rewards, all_lengths, all_games = [], [], []
        
        obs = self.eval_env.reset()

        while True:
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.eval_env.step(actions)
            for i, info in enumerate(infos):
                if 'episode' in info:
                    all_rewards.append(info['episode']['r'])
                    all_lengths.append(info['episode']['l'])
                    all_games.append(info['game_name'])
                    episodes_counts[info['game_name']] += 1
                    print(episodes_counts[info['game_name']])

            # Check if all games reached n_eval_episodes -> Now we have at least n_eval_episodes for each game
            if all(count >= self.n_eval_episodes for count in episodes_counts.values()):
                break

        # Aggregate results per game
        game_stats = {}
        for game in self.eval_game_names:
            game_indices = [i for i, g in enumerate(all_games) if g == game][:self.n_eval_episodes]
            rewards = [all_rewards[i] for i in game_indices]
            lengths = [all_lengths[i] for i in game_indices]
            game_stats[game] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_length': np.mean(lengths)
            }
        return game_stats

class OwnLoggingCallback(BaseCallback):
    """
    Custom callback for logging training metrics (reward, episode length, local steps)
    during the training of individual games in normal curriculum.
    Also logs the current learning rate at the end of each rollout.
    """
    def __init__(self, game_name, global_offset: int =0, verbose=0):
        super().__init__(verbose)
        self.game_name = game_name
        self.global_offset = global_offset

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])  
        for info in infos:
            if "episode" in info:
                # Episode reward and length is collected
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                local_steps[self.game_name] += ep_length

                global_step = self.global_offset + self.num_timesteps
            
                wandb.log({
                    f"{self.game_name}/episode_reward": ep_reward,
                    f"{self.game_name}/episode_length": ep_length,
                    f"{self.game_name}/local_steps": local_steps[self.game_name], 
                }, step=global_step)  # SB3's timestep
                print(f"[{self.game_name}] Step: {global_step} | Reward: {ep_reward} | Episode Length: {ep_length}")
        return True

    def _on_rollout_end(self):
        """
        Logs the current learning rate after each rollout of current game.
        """
        current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        global_step = self.global_offset + self.num_timesteps
        wandb.log({
            f"{self.game_name}/lr": current_lr,
        }, step=global_step)
        return


def main_training_loop_nc(games_list_train, global_iterations):
    """
    This is where the main training across multiple global iterations for the normal curriculum strategy happens.
    Iterates through training games in sequence, logging offsets for correct evaluation.

    Args:
        games_list_train (List[str]): Names of the games
        global_iterations (int): variable to keep track of how many blocks of training where done
    """
    global_train_offset = 0

    for glo_i in range(global_iterations+1):
        if glo_i == 0:
            continue
        games = config['train_games']
        local_i = 1
        for game in games:
            print(f"Training on: {game}")
            if local_i == 1:
                model_steps= train_game(game_name=game, local_i=local_i, prev_game_name=games[local_i], global_i=glo_i, FirstGlobalIter= True, global_train_offset=global_train_offset) 
            else: 
                model_steps=train_game(game_name=game, local_i=local_i, prev_game_name=games[local_i-2], global_i=glo_i, global_train_offset=global_train_offset) 
            local_i += 1

            global_train_offset += model_steps

def train_game(game_name, local_i, prev_game_name, global_i, FirstGlobalIter=False, global_train_offset = 0):
    """
    Core training function for one game in the normal curriculum.

    Loads previous model (if not first iteration), initializes the training environment,
    constructs callbacks, and runs PPO training. Saves model after training.

    Args:
        game_name (str): Current game being trained.
        local_i (int): Index of the local curriculum step.
        prev_game_name (str): Previous game for loading model.
        global_i (int): Global curriculum iteration.
        FirstGlobalIter (bool): Flag indicating if it's the first global iteration.
        global_train_offset (int): Timestep offset used for logging.
    """

    model_dir = "./models/normal_curriculum/"
    if local_i not in [1, 2, 3]:
        raise ValueError("Iteration must be 1, 2, or 3!!!")

    dir_tmp_save = f"{game_name}_{config['project_name']}_{config['run_name']}"

    model_save_path = model_dir + f"ppo_glo{global_i}_l{local_i}{dir_tmp_save}"

    time_start = time.time()

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    env_game = SubprocVecEnv([
            env_custom_wrapper(game_name, seed=config["seed"] + i) 
            for i in range(config["n_envs"])
            ])  

    all_envs = config["train_games"] + [config["eval_game"]]
    eval_env = make_evaluation_envs(all_envs, base_seed = config["seed"])

    """
    make All Callbacks
    1. Own For Training Metrics
    2. Multi_Eval To evaluate all 4 enviroments in parallel
    """
    own_callback = OwnLoggingCallback(game_name, global_offset=global_train_offset, verbose=1)

    multi_eval_callback = MultiEvalCallback(
        eval_env=eval_env,
        eval_game_names=all_envs,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        global_offset=global_train_offset,
        verbose=1
    )

    initial_lr = config['lr'] * lr_sched[global_i -1]  # Start from previous block's end LR
    final_lr = config['lr'] * lr_sched[global_i]  # Decay to the next LR in schedule
    learning_rate_schedule = linear_schedule_for_lr(initial_lr, final_lr)
    print("Observation Space:", env_game.observation_space)
    print("Observation Shape:", env_game.observation_space.shape)
    
    if local_i == 1 and global_i ==1 :   
        model = PPO(
            policy="CnnPolicy",
            env=env_game,  # Ensure env_game is defined appropriately
            verbose=1,
            tensorboard_log=f"./tb_logs/{config['project_name']}/{config['run_name']}",
            learning_rate = learning_rate_schedule,
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            n_epochs=config["epochs"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            device=config["device"],
            seed = config["seed"]
        )
    else:
        if FirstGlobalIter== True:
            tmp_global = global_i -1
            tmp_iter = len(config["train_games"])
            tmp_game_name = config["train_games"][len(config["train_games"])-1]
            model = PPO.load(f"./models/normal_curriculum/ppo_glo{tmp_global}_l{tmp_iter}{tmp_game_name}_{config['project_name']}_{config['run_name']}.zip")
            FirstGlobalIter = False
            
        else:    
            tmp_iter = local_i - 1 # To get the model from last iteration
            model = PPO.load(f"./models/normal_curriculum/ppo_glo{global_i}_l{tmp_iter}{prev_game_name}_{config['project_name']}_{config['run_name']}.zip")
            
        model.set_env(env_game)
        
    # Set new LR
    model.lr_schedule = learning_rate_schedule
    model.learning_rate = model.lr_schedule  
    model._current_progress_remaining = 1.0

    model.learn(
        total_timesteps=config['timesteps_per_game'],
        tb_log_name=f"ppo_glo{global_i}_l{local_i}{game_name}{config['run_name']}",
        callback=[
            own_callback,                                              
            multi_eval_callback
        ]
    )

    model.save(model_save_path)
    time_game_elapsed(time_start, game_name)
    env_game.close()

    return model.num_timesteps


"""
normal_curriulum methods above
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
random_curriulum stuff below
"""


class RandomGameSwitchEnv(gym.Env):
    """
    A custom Gym environment that dynamically switches between multiple games during training.

    This environment ensures balanced training by prioritizing the game with the least steps so far.
    Game switching happens at the beginning of each episode.
    """
    def __init__(self, game_names):
        super().__init__()
        self.game_names = game_names
        # Pre-create all environments during initialization
        self.game_envs = {game: env_custom_wrapper(game)() for game in game_names}
        self.current_env = None
        self.current_game = None
        self.game_steps = {game: 0 for game in game_names}

        # Initialize with a random game to set initial spaces
        self.current_game = random.choice(game_names)
        self.current_env = self.game_envs[self.current_game]
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def reset(self, **kwargs):
        # Select the game with the least steps
        min_steps = min(self.game_steps.values())
        candidates = [g for g in self.game_names if self.game_steps[g] == min_steps]
        self.current_game = random.choice(candidates)
        
        # Switch to the pre-created environment and reset it
        self.current_env = self.game_envs[self.current_game]
        obs, info = self.current_env.reset(**kwargs)
        info['game_name'] = self.current_game
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.current_env.step(action)
        info['game_name'] = self.current_game
        if done or truncated:
            if "episode" in info:
                ep_len = info["episode"].get("l", 0)
                self.game_steps[self.current_game] += ep_len
        return obs, reward, done, truncated, info

    def close(self):
        # Close all pre-created environments
        for env in self.game_envs.values():
            env.close()

class RandomLoggingCallback(BaseCallback):
    """
    A callback that logs training metrics per game.
    It reads the 'game_name' key in the info dict (attached by RandomGameSwitchEnv)
    to store episode rewards and lengths.
    Also logs the learning rate at the end of each rollout.
    """
    def __init__(self, game_names, verbose=1):
        super().__init__(verbose)
        self.game_names = game_names  # List of training game names

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                game = info.get("game_name", "unknown")
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                # Update per-game step counter.
                local_steps[game] += ep_length
                step = local_steps[game]
                
                wandb.log({
                    f"{game}/episode_reward": ep_reward,
                    f"{game}/episode_length": ep_length,
                    f"{game}/local_steps": local_steps[game], 
                }, step=self.num_timesteps)  # Use SB3's timestep
                print(f"[{game}] Step: {step} | Reward: {ep_reward} | Episode Length: {ep_length}")
        return True

    def _on_rollout_end(self):
        # Log the current learning rate after each rollout.
        current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        for game in self.game_names:
            step = local_steps.get(game, 0)
            wandb.log({
                f"{game}/lr": current_lr,
            }, step=self.num_timesteps)
        return

def make_random_env(game_names):
    def _init():
        return RandomGameSwitchEnv(game_names)
    return _init

def train_random_curriculum(config):
    """
    Trains PPO on a random curriculum: at each episode reset a random training game.
    Also performs periodic evaluation on the held-out eval game.
    """
    env_train = SubprocVecEnv([make_random_env(config["train_games"]) for _ in range(config['n_envs'])])

    initial_lr = config['lr'] * 1  # Start from previous block's end LR
    final_lr = config['lr'] * 0  # Decay to the next LR in schedule
    learning_rate_schedule = linear_schedule_for_lr(initial_lr, final_lr)

    model = PPO(
            policy="CnnPolicy",
            env=env_train,
            verbose=1,
            tensorboard_log=f"./tb_logs/{config['project_name']}/{config['run_name']}",
            learning_rate=learning_rate_schedule,
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            n_epochs=config["epochs"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            device=config["device"],
        )

    """
    Create callbacks: one for logging training metrics and one for combined evaluation environment
    """
    random_logging_callback = RandomLoggingCallback(game_names=config["train_games"], verbose=1)
    all_envs = config["train_games"] + [config["eval_game"]]
    eval_env = make_evaluation_envs(all_envs, config["seed"])
    multi_eval_callback = MultiEvalCallback(
        eval_env=eval_env,
        eval_game_names=all_envs,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        verbose=1
    )

    model.set_env(env_train)

    model.learn(total_timesteps=config['total_timesteps'],callback=[
            random_logging_callback,                                              
            multi_eval_callback
        ])


    env_train.close()

if __name__ == "__main__":
    seeds = [42, 137, 420, 666, 999]  # 5 different seeds for 5 runs

    for seed in seeds:
        config = {
            "project_name": "Binary_NC_Seeded",
            "tag": "Binary_NC",   
            "run_name": "SI_P",
            "notes": "Normal Curriculum. Eval_freq 30000. Loop for making 3 runs",
            "train_games": [ "SpaceInvaders", "Phoenix"],
            "1st_game":  "SpaceInvaders",
            "2nd_game": "Phoenix",
            "3rd_game": "-",
            "eval_game": "Galaxian",
            "rcornc": "nc",     # wether to use normal_curr (nc) or block_curriculum (bc) as training regime (if baseline just select nc and "num_blocks": 1)
            "loops_anzahl": 2,
            "nr_train_games": 2,
            "num_blocks": 2,
            "timesteps_per_game":5_000_000,  # importnatn for nc
            "total_timesteps": 10_000_000, # imporntatn for rc
            "ocorgym": "oc", 
            "n_envs": 8,
            "eval_freq": 30000,
            "lr": 0.00025,
            "epochs": 4,
            "n_steps": 128,
            "gamma": 0.99,
            "batch_size": 256,
            "epochs": 4,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "device": "cuda",
            "n_eval_episodes": 5,
            "ocatari_obs_mode": "dqn",
            "ocatari_mode": "ram",
            "ocatari_buffer_size": 4,
            "oc_wrapper": "binary", #which oc_wrapper to use: bianry, planes, none
            "ocwrapper_pixels": False,
            "ocwrapper_buffer_size": 4,
            "ocwrapper_work_in_output_shape": True,
            "seed": seed
        }


    for i in range(config["loops_anzahl"]):

        global_train_offset = 0

        wandb.init(
            project=config['project_name'],
            name=config['run_name'],
            config=config,
            notes=config['notes']
        )

        lr_sched =lr_schedule(config['num_blocks'])

        """
        Either random_curriuclum (rc) or normal_curriuclum (nc) training regime training starts
        """
        # start training nc
        if config['rcornc'] == 'nc':
            main_training_loop_nc(games_list_train= config['train_games'], global_iterations= config['num_blocks'])

        # start training using rc
        elif config['rcornc'] == 'rc':
            train_random_curriculum(config)  

        else:
            print("There was not a correct training regime selected, pelase selectred either 'rc' for random_curriulum or 'nc' for normal_curriulum")

        wandb.finish()