import os
import yaml
from typing import Dict, Any
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import torch
from datetime import datetime

from environments.traffic_env import MultiAgentTrafficEnv
from utils.training_logger import TrainingLogger
from utils.ray_callbacks import MetricsLoggerCallback


class MAPPOTrainer:
    
    def __init__(self, 
                 sumo_config_path: str = "config/sumo_config.yaml",
                 training_config_path: str = "config/training_config.yaml"):
        
        with open(sumo_config_path, 'r') as f:
            self.sumo_config = yaml.safe_load(f)
        
        with open(training_config_path, 'r') as f:
            self.training_config = yaml.safe_load(f)['training']
        
        self.checkpoint_dir = os.path.abspath("checkpoints")
        self.results_dir = os.path.abspath("results")
        self.logs_dir = os.path.abspath("logs")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"MAPPO_{timestamp}"
        self.logger = TrainingLogger(log_dir=self.logs_dir, experiment_name=experiment_name)
        
        if not ray.is_initialized():
            ray.init(
                num_gpus=self.training_config.get('num_gpus', 1),
                ignore_reinit_error=True,
                log_to_driver=True
            )
        
        self._register_environment()
        
        print(f"🚦 Entrenador MAPPO inicializado")
        print(f"   GPU disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def _register_environment(self):
        def env_creator(config):
            import sumo_rl
            import numpy as np
            from gymnasium import spaces
            
            sumo_config = config.get('sumo', {})
            
            sumo_params = {
                'net_file': sumo_config.get('net_file', 'scenarios/simple_grid/grid.net.xml'),
                'route_file': sumo_config.get('route_file', 'scenarios/simple_grid/grid.rou.xml'),
                'use_gui': sumo_config.get('use_gui', False),
                'num_seconds': sumo_config.get('num_seconds', 3600),
                'delta_time': sumo_config.get('delta_time', 5),
                'yellow_time': sumo_config.get('yellow_time', 2),
                'min_green': sumo_config.get('min_green', 5),
                'max_green': sumo_config.get('max_green', 60),
                'single_agent': False,
                'sumo_seed': sumo_config.get('sumo_seed', 42),
                'fixed_ts': sumo_config.get('fixed_ts', False),
            }
            
            reward_fn = sumo_config.get('reward_fn', 'diff-waiting-time')
            if reward_fn:
                sumo_params['reward_fn'] = reward_fn
            
            pettingzoo_env = sumo_rl.parallel_env(**sumo_params)
            
            pettingzoo_env.reset()
            max_obs_size = max(
                pettingzoo_env.observation_space(agent).shape[0] 
                for agent in pettingzoo_env.possible_agents
            )
            
            class PaddedObsWrapper:
                def __init__(self, env, max_size):
                    self.env = env
                    self.max_size = max_size
                    self.possible_agents = env.possible_agents
                
                @property
                def agents(self):
                    return self.env.agents
                    
                def observation_space(self, agent):
                    return spaces.Box(
                        low=-np.inf, 
                        high=np.inf, 
                        shape=(self.max_size,), 
                        dtype=np.float32
                    )
                
                def action_space(self, agent):
                    return self.env.action_space(agent)
                
                def _pad_obs(self, obs):
                    if len(obs) < self.max_size:
                        return np.pad(obs, (0, self.max_size - len(obs)), 'constant')
                    return obs
                
                def reset(self, **kwargs):
                    obs_dict, info = self.env.reset(**kwargs)
                    return {agent: self._pad_obs(obs) for agent, obs in obs_dict.items()}, info
                
                def step(self, actions):
                    obs_dict, rewards, terms, truncs, infos = self.env.step(actions)
                    padded_obs = {agent: self._pad_obs(obs) for agent, obs in obs_dict.items()}
                    return padded_obs, rewards, terms, truncs, infos
                
                def close(self):
                    self.env.close()
            
            wrapped_env = PaddedObsWrapper(pettingzoo_env, max_obs_size)
            return ParallelPettingZooEnv(wrapped_env)
        
        register_env("traffic_env", env_creator)
    
    def _create_config(self) -> PPOConfig:
        env_config = {
            'sumo': self.sumo_config['sumo']
        }
        
        import sumo_rl
        sumo_config = self.sumo_config['sumo']
        
        sumo_params = {
            'net_file': sumo_config.get('net_file', 'scenarios/simple_grid/grid.net.xml'),
            'route_file': sumo_config.get('route_file', 'scenarios/simple_grid/grid.rou.xml'),
            'use_gui': False,
            'num_seconds': sumo_config.get('num_seconds', 3600),
            'delta_time': sumo_config.get('delta_time', 5),
            'yellow_time': sumo_config.get('yellow_time', 2),
            'min_green': sumo_config.get('min_green', 5),
            'max_green': sumo_config.get('max_green', 60),
            'single_agent': False,
            'sumo_seed': sumo_config.get('sumo_seed', 42),
        }
        
        reward_fn = sumo_config.get('reward_fn', 'diff-waiting-time')
        if reward_fn:
            sumo_params['reward_fn'] = reward_fn
        
        temp_env = sumo_rl.parallel_env(**sumo_params)
        agent_ids = temp_env.possible_agents
        
        temp_env.reset()
        max_obs_size = max(
            temp_env.observation_space(agent).shape[0] 
            for agent in agent_ids
        )
        
        sample_agent = agent_ids[0]
        action_space = temp_env.action_space(sample_agent)
        
        import numpy as np
        from gymnasium import spaces
        obs_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(max_obs_size,), 
            dtype=np.float32
        )
        
        temp_env.close()
        
        config = (
            PPOConfig()
            .environment(
                env="traffic_env",
                env_config=env_config,
            )
            .framework(self.training_config.get('framework', 'torch'))
            .resources(
                num_gpus=self.training_config.get('num_gpus', 1),
                num_cpus_per_worker=1,
            )
            .rollouts(
                num_rollout_workers=self.training_config.get('num_workers', 4),
                rollout_fragment_length=self.training_config.get('rollout_fragment_length', 200),
                batch_mode=self.training_config.get('batch_mode', 'truncate_episodes'),
            )
            .training(
                train_batch_size=self.training_config.get('train_batch_size', 4000),
                sgd_minibatch_size=self.training_config.get('sgd_minibatch_size', 128),
                num_sgd_iter=self.training_config.get('num_sgd_iter', 10),
                lr=self.training_config.get('lr', 0.0003),
                gamma=self.training_config.get('gamma', 0.99),
                lambda_=self.training_config.get('lambda', 0.95),
                clip_param=self.training_config.get('clip_param', 0.2),
                vf_clip_param=self.training_config.get('vf_clip_param', 10.0),
                entropy_coeff=self.training_config.get('entropy_coeff', 0.01),
                model={
                    "fcnet_hiddens": self.training_config['model'].get('fcnet_hiddens', [256, 256]),
                    "fcnet_activation": self.training_config['model'].get('fcnet_activation', 'relu'),
                    "vf_share_layers": self.training_config['model'].get('vf_share_layers', False),
                },
            )
            .multi_agent(
                policies={"shared_policy": (None, obs_space, action_space, {})},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .evaluation(
                evaluation_interval=self.training_config['evaluation'].get('evaluation_interval', 10),
                evaluation_duration=self.training_config['evaluation'].get('evaluation_duration', 10),
                evaluation_num_workers=self.training_config['evaluation'].get('evaluation_num_workers', 1),
                evaluation_config=self.training_config['evaluation'].get('evaluation_config', {}),
            )
        )
        
        return config
    
    def train(self, num_iterations: int = None):
        if num_iterations is None:
            num_iterations = self.training_config['stopping_criteria'].get('training_iteration', 500)
        
        config = self._create_config()
        
        print(f"\n🚀 Iniciando entrenamiento MAPPO")
        print(f"   Iteraciones: {num_iterations}")
        print(f"   Workers: {self.training_config.get('num_workers', 4)}")
        print(f"   Batch size: {self.training_config.get('train_batch_size', 4000)}")
        print(f"   Learning rate: {self.training_config.get('lr', 0.0003)}")
        print(f"\n{'='*60}\n")
        
        metrics_callback = MetricsLoggerCallback(self.logger)
        
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=ray.air.RunConfig(
                name="MAPPO_TrafficSignals",
                storage_path=self.results_dir,
                stop={
                    "training_iteration": num_iterations,
                },
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=self.training_config['checkpoint'].get('checkpoint_freq', 10),
                    checkpoint_at_end=self.training_config['checkpoint'].get('checkpoint_at_end', True),
                    num_to_keep=self.training_config['checkpoint'].get('keep_checkpoints_num', 5),
                ),
                callbacks=[metrics_callback],
                verbose=1,
            ),
        )
        
        results = tuner.fit()
        
        try:
            best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        except:
            best_result = results.get_best_result()
        
        print(f"\n{'='*60}")
        print(f"✅ Entrenamiento completado!")
        
        reward_keys = ['episode_reward_mean', 'env_runners/episode_reward_mean', 
                       'sampler_results/episode_reward_mean']
        reward_value = None
        for key in reward_keys:
            if key in best_result.metrics:
                reward_value = best_result.metrics[key]
                break
        
        if reward_value is not None:
            print(f"   Mejor recompensa promedio: {reward_value:.2f}")
        
        if best_result.checkpoint:
            print(f"   Checkpoint guardado en: {best_result.checkpoint.path}")
        
        print(f"{'='*60}\n")
        
        return results, best_result
    
    def load_checkpoint(self, checkpoint_path: str):
        config = self._create_config()
        algo = config.build()
        algo.restore(checkpoint_path)
        return algo
    
    def evaluate(self, checkpoint_path: str, num_episodes: int = 10):
        print(f"\n📊 Evaluando modelo desde: {checkpoint_path}")
        
        algo = self.load_checkpoint(checkpoint_path)
        
        env_config = {
            'sumo': self.sumo_config['sumo']
        }
        env = MultiAgentTrafficEnv(env_config)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            observations = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                actions = {}
                for agent_id, obs in observations.items():
                    action = algo.compute_single_action(
                        obs, 
                        policy_id="shared_policy",
                        explore=False
                    )
                    actions[agent_id] = action
                
                observations, rewards, dones, _ = env.step(actions)
                episode_reward += sum(rewards.values())
                done = dones['__all__']
            
            episode_rewards.append(episode_reward)
            print(f"   Episodio {episode + 1}/{num_episodes}: Recompensa = {episode_reward:.2f}")
        
        env.close()
        algo.stop()
        
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\n   Recompensa promedio: {avg_reward:.2f}")
        print(f"   Recompensa máxima: {max(episode_rewards):.2f}")
        print(f"   Recompensa mínima: {min(episode_rewards):.2f}")
        
        return episode_rewards
    
    def shutdown(self):
        if ray.is_initialized():
            ray.shutdown()

