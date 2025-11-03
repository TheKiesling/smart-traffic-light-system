import os
import sys
from typing import Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sumo_rl


class TrafficSignalEnv(gym.Env):
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        self.sumo_config = self.config.get('sumo', {})
        self.env_config = self.config.get('environment', {})
        
        self._setup_sumo_paths()
        
        sumo_params = {
            'net_file': self.sumo_config.get('net_file', 'scenarios/simple_grid/grid.net.xml'),
            'route_file': self.sumo_config.get('route_file', 'scenarios/simple_grid/grid.rou.xml'),
            'use_gui': self.sumo_config.get('use_gui', False),
            'num_seconds': self.sumo_config.get('num_seconds', 3600),
            'delta_time': self.sumo_config.get('delta_time', 5),
            'yellow_time': self.sumo_config.get('yellow_time', 2),
            'min_green': self.sumo_config.get('min_green', 5),
            'max_green': self.sumo_config.get('max_green', 60),
            'single_agent': self.sumo_config.get('single_agent', False),
            'sumo_seed': self.sumo_config.get('sumo_seed', 42),
            'fixed_ts': self.sumo_config.get('fixed_ts', False),
        }
        
        reward_fn = self.sumo_config.get('reward_fn', 'diff-waiting-time')
        if reward_fn:
            sumo_params['reward_fn'] = reward_fn
        
        self.env = sumo_rl.parallel_env(**sumo_params)
        
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        
        self.env.reset()
        sample_obs, _ = self.env.last()
        
        if isinstance(sample_obs, dict):
            obs_shape = list(sample_obs.values())[0].shape[0]
        else:
            obs_shape = sample_obs.shape[0]
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )
        
        sample_agent = self.agents[0]
        self.action_space = self.env.action_space(sample_agent)
        
        self._agent_ids = set(self.agents)
        
    def _setup_sumo_paths(self):
        if 'SUMO_HOME' not in os.environ:
            if sys.platform.startswith('win'):
                sumo_paths = [
                    'C:/Program Files (x86)/Eclipse/Sumo',
                    'C:/Program Files/Eclipse/Sumo',
                ]
                for path in sumo_paths:
                    if os.path.exists(path):
                        os.environ['SUMO_HOME'] = path
                        break
            else:
                os.environ['SUMO_HOME'] = '/usr/share/sumo'
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if tools not in sys.path:
                sys.path.append(tools)
    
    def reset(self, *, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        
        obs_array = np.array(list(observations.values())[0], dtype=np.float32)
        
        return obs_array, infos
    
    def step(self, action):
        action_dict = {agent: action for agent in self.agents}
        
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        
        obs_array = np.array(list(observations.values())[0], dtype=np.float32)
        reward = sum(rewards.values()) / len(rewards)
        
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        
        return obs_array, reward, terminated, truncated, infos
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def render(self):
        pass


class MultiAgentTrafficEnv:
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sumo_config = self.config.get('sumo', {})
        
        self._setup_sumo_paths()
        
        sumo_params = {
            'net_file': self.sumo_config.get('net_file', 'scenarios/simple_grid/grid.net.xml'),
            'route_file': self.sumo_config.get('route_file', 'scenarios/simple_grid/grid.rou.xml'),
            'use_gui': self.sumo_config.get('use_gui', False),
            'num_seconds': self.sumo_config.get('num_seconds', 3600),
            'delta_time': self.sumo_config.get('delta_time', 5),
            'yellow_time': self.sumo_config.get('yellow_time', 2),
            'min_green': self.sumo_config.get('min_green', 5),
            'max_green': self.sumo_config.get('max_green', 60),
            'single_agent': False,
            'sumo_seed': self.sumo_config.get('sumo_seed', 42),
            'fixed_ts': self.sumo_config.get('fixed_ts', False),
        }
        
        reward_fn = self.sumo_config.get('reward_fn', 'diff-waiting-time')
        if reward_fn:
            sumo_params['reward_fn'] = reward_fn
        
        self.env = sumo_rl.parallel_env(**sumo_params)
        
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        self._agent_ids = set(self.agents)
        
        if self.agents:
            sample_agent = self.agents[0]
            self.observation_space = self.env.observation_space(sample_agent)
            self.action_space = self.env.action_space(sample_agent)
        else:
            self.observation_space = None
            self.action_space = None
        
    def _setup_sumo_paths(self):
        if 'SUMO_HOME' not in os.environ:
            if sys.platform.startswith('win'):
                sumo_paths = [
                    'C:/Program Files (x86)/Eclipse/Sumo',
                    'C:/Program Files/Eclipse/Sumo',
                ]
                for path in sumo_paths:
                    if os.path.exists(path):
                        os.environ['SUMO_HOME'] = path
                        break
            else:
                os.environ['SUMO_HOME'] = '/usr/share/sumo'
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if tools not in sys.path:
                sys.path.append(tools)
    
    def reset(self, *, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations
    
    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        
        dones = {
            agent: terminations.get(agent, False) or truncations.get(agent, False)
            for agent in self.agents
        }
        dones['__all__'] = all(dones.values())
        
        return observations, rewards, dones, infos
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()

