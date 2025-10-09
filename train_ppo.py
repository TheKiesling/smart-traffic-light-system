# train_ppo.py
import os
import argparse
import datetime as dt

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from sb3_env_adapter import SB3TrafficEnv

def make_env(model_path: str, seed: int = 0):
    def _thunk():
        env = SB3TrafficEnv(model_path)
        # Monitor registra reward/episodios para TensorBoard
        env = Monitor(env)
        return env
    return _thunk

def main(args):
    os.environ.setdefault("JAVA_EXE", r"C:\Program Files\Java\jdk-17\bin\java.exe")

    run_name = dt.datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(args.log_dir or "./runs")
    save_dir = os.path.abspath(args.save_dir or "./checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Entrenaremos con 1 env por seguridad (AnyLogic es “pesado”)
    vec_env = DummyVecEnv([make_env(args.model_path, seed=0)])
    vec_env = VecMonitor(vec_env, filename=None)

    # Entorno de evaluación (se lanza otra instancia del modelo)
    eval_env = DummyVecEnv([make_env(args.model_path, seed=123)])
    eval_env = VecMonitor(eval_env, filename=None)

    # Modelo PPO base
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,          # batch “on-policy” antes de cada update
        batch_size=256,
        n_epochs=10,           # épocas por update (no “episodios”)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
        seed=0,
    )

    # Callbacks: checkpoints + evaluación periódica
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,              # guarda cada 20k timesteps
        save_path=save_dir,
        name_prefix=run_name
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000,              # eval cada 10k timesteps
        n_eval_episodes=3,             # eval corta (AnyLogic cuesta)
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([ckpt_cb, eval_cb])

    # Entrenamiento
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Guarda el modelo final
    model.save(os.path.join(save_dir, f"{run_name}_final"))

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r"C:\al_model\model.jar")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    main(args)
