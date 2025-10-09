# evaluate_policy.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_env_adapter import SB3TrafficEnv

MODEL_PATH = r"C:\al_model\model.jar"
POLICY_PATH = r".\checkpoints\best_model.zip"

def make_env():
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(SB3TrafficEnv(MODEL_PATH))
    return _thunk

if __name__ == "__main__":
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env, filename=None)
    model = PPO.load(POLICY_PATH, env=eval_env)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f"Mean reward over 5 eps: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()
