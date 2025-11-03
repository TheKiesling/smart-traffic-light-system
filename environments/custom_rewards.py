import numpy as np
from typing import Dict


class TimeConstrainedReward:    
    def __init__(self, max_green: int = 60, min_green: int = 5):
        self.max_green = max_green
        self.min_green = min_green
        self.phase_times = {}
        
    def __call__(self, ts) -> float:
        ts_id = ts.id
        current_phase = ts.green_phase
        
        if ts_id not in self.phase_times:
            self.phase_times[ts_id] = {
                'last_phase': current_phase,
                'time_in_phase': 0
            }
        
        info = self.phase_times[ts_id]
        
        if info['last_phase'] == current_phase:
            info['time_in_phase'] += ts.delta_time
        else:
            info['time_in_phase'] = ts.delta_time
            info['last_phase'] = current_phase
        
        time_in_green = info['time_in_phase']
        
        diff_waiting_time = ts.get_accumulated_waiting_time_per_lane()
        reward = -sum(diff_waiting_time)
        
        if time_in_green > self.max_green:
            penalty = (time_in_green - self.max_green) * 2.0
            reward -= penalty
        
        elif time_in_green < self.min_green:
            penalty = (self.min_green - time_in_green) * 1.5
            reward -= penalty
        
        return reward


class HybridReward:   
    def __init__(self, 
                 max_green: int = 60, 
                 min_green: int = 5,
                 weight_waiting: float = 1.0,
                 weight_queue: float = 0.5,
                 weight_throughput: float = 0.3,
                 weight_time_penalty: float = 2.0):
        
        self.max_green = max_green
        self.min_green = min_green
        self.weight_waiting = weight_waiting
        self.weight_queue = weight_queue
        self.weight_throughput = weight_throughput
        self.weight_time_penalty = weight_time_penalty
        
        self.phase_times = {}
        self.last_departed = {}
        
    def __call__(self, ts) -> float:
        ts_id = ts.id
        current_phase = ts.green_phase
        
        if ts_id not in self.phase_times:
            self.phase_times[ts_id] = {
                'last_phase': current_phase,
                'time_in_phase': 0
            }
            self.last_departed[ts_id] = 0
        
        info = self.phase_times[ts_id]
        
        if info['last_phase'] == current_phase:
            info['time_in_phase'] += ts.delta_time
        else:
            info['time_in_phase'] = ts.delta_time
            info['last_phase'] = current_phase
        
        time_in_green = info['time_in_phase']
        
        waiting_time = sum(ts.get_accumulated_waiting_time_per_lane())
        queue_length = sum(ts.get_lanes_queue())
        
        departed = ts.env.sim.vehicle_count_departedIDs
        throughput = len(departed) - self.last_departed[ts_id]
        self.last_departed[ts_id] = len(departed)
        
        reward = 0.0
        
        reward -= self.weight_waiting * waiting_time
        reward -= self.weight_queue * queue_length
        reward += self.weight_throughput * throughput
        
        if time_in_green > self.max_green:
            penalty = (time_in_green - self.max_green) / self.max_green
            reward -= self.weight_time_penalty * penalty * 100
        
        elif time_in_green < self.min_green:
            penalty = (self.min_green - time_in_green) / self.min_green
            reward -= self.weight_time_penalty * penalty * 50
        
        return reward


def get_reward_function(name: str, **kwargs):
    if name == 'time-constrained':
        return TimeConstrainedReward(**kwargs)
    
    elif name == 'hybrid':
        return HybridReward(**kwargs)
    
    elif name in ['diff-waiting-time', 'average-speed', 'queue', 'pressure']:
        return name
    
    else:
        raise ValueError(f"Función de recompensa desconocida: {name}")


