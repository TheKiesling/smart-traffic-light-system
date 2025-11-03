import argparse
import os
import sys
import yaml
from datetime import datetime

from models.mappo_trainer import MAPPOTrainer
from utils.helpers import (
    print_system_info, 
    check_sumo_installation,
    plot_training_metrics
)


def train(args):    
    print_system_info()
    
    if not check_sumo_installation():
        if not args.force:
            sys.exit(1)
    
    trainer = MAPPOTrainer(
        sumo_config_path=args.sumo_config,
        training_config_path=args.training_config
    )
    
    try:
        results, best_result = trainer.train(num_iterations=args.iterations)
        
        if best_result and best_result.checkpoint:
            print(f"\n💾 Checkpoint guardado en: {best_result.checkpoint.path}")
        
        if args.plot and best_result:
            try:
                plot_path = os.path.join("results", "training_metrics.png")
                plot_training_metrics(best_result.log_dir, save_path=plot_path)
            except Exception as plot_error:
                print(f"⚠️  No se pudieron generar las gráficas: {plot_error}")
        
    except Exception as e:
        raise
    finally:
        trainer.shutdown()


def evaluate(args):    
    if not os.path.exists(args.checkpoint):
        sys.exit(1)
    
    trainer = MAPPOTrainer(
        sumo_config_path=args.sumo_config,
        training_config_path=args.training_config
    )
    
    try:
        rewards = trainer.evaluate(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes
        )
        
    except Exception as e:
        raise
    finally:
        trainer.shutdown()


def visualize(args):    
    if not os.path.exists(args.checkpoint):
        sys.exit(1)
    

    with open(args.sumo_config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['sumo']['use_gui'] = True
    config['sumo']['num_seconds'] = args.duration
    
    from models.mappo_trainer import MAPPOTrainer
    
    trainer = MAPPOTrainer(
        sumo_config_path=args.sumo_config,
        training_config_path=args.training_config
    )
    
    algo = trainer.load_checkpoint(args.checkpoint)
    
    import sumo_rl
    import numpy as np
    from gymnasium import spaces
    
    sumo_params = {
        'net_file': config['sumo']['net_file'],
        'route_file': config['sumo']['route_file'],
        'use_gui': True,
        'num_seconds': args.duration,
        'delta_time': config['sumo'].get('delta_time', 5),
        'yellow_time': config['sumo'].get('yellow_time', 2),
        'min_green': config['sumo'].get('min_green', 5),
        'max_green': config['sumo'].get('max_green', 60),
        'single_agent': False,
        'sumo_seed': config['sumo'].get('sumo_seed', 42),
        'fixed_ts': config['sumo'].get('fixed_ts', False),
    }
    
    reward_fn = config['sumo'].get('reward_fn', 'diff-waiting-time')
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
    
    env = PaddedObsWrapper(pettingzoo_env, max_obs_size)
    
    try:
        observations, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while env.agents:
            actions = {}
            for agent_id, obs in observations.items():
                action = algo.compute_single_action(
                    obs, 
                    policy_id="shared_policy",
                    explore=False
                )
                actions[agent_id] = action
            
            observations, rewards, terms, truncs, _ = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"   Paso {step_count}: Recompensa acumulada = {total_reward:.2f}")
        
        print(f"\n✅ Simulación completada")
        print(f"   Recompensa total: {total_reward:.2f}")
        
    except Exception as e:
        raise
    finally:
        env.close()
        algo.stop()
        trainer.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Semáforos Inteligentes con MAPPO y SUMO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  Entrenar el modelo:
    python main.py train --iterations 100
  
  Evaluar un modelo entrenado:
    python main.py evaluate --checkpoint checkpoints/checkpoint_000100
  
  Visualizar con SUMO-GUI:
    python main.py visualize --checkpoint checkpoints/checkpoint_000100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    train_parser = subparsers.add_parser('train', help='Entrenar el modelo MAPPO')
    train_parser.add_argument('--iterations', type=int, default=None,
                             help='Número de iteraciones de entrenamiento')
    train_parser.add_argument('--sumo-config', type=str, default='config/sumo_config.yaml',
                             help='Ruta al archivo de configuración de SUMO')
    train_parser.add_argument('--training-config', type=str, default='config/training_config.yaml',
                             help='Ruta al archivo de configuración de entrenamiento')
    train_parser.add_argument('--plot', action='store_true',
                             help='Generar gráficas al finalizar')
    train_parser.add_argument('--force', action='store_true',
                             help='Continuar aunque SUMO no esté instalado')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluar un modelo entrenado')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Ruta al checkpoint del modelo')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Número de episodios para evaluar')
    eval_parser.add_argument('--sumo-config', type=str, default='config/sumo_config.yaml',
                            help='Ruta al archivo de configuración de SUMO')
    eval_parser.add_argument('--training-config', type=str, default='config/training_config.yaml',
                            help='Ruta al archivo de configuración de entrenamiento')
    
    viz_parser = subparsers.add_parser('visualize', help='Visualizar con SUMO-GUI')
    viz_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Ruta al checkpoint del modelo')
    viz_parser.add_argument('--duration', type=int, default=3600,
                           help='Duración de la simulación en segundos')
    viz_parser.add_argument('--sumo-config', type=str, default='config/sumo_config.yaml',
                           help='Ruta al archivo de configuración de SUMO')
    viz_parser.add_argument('--training-config', type=str, default='config/training_config.yaml',
                           help='Ruta al archivo de configuración de entrenamiento')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'visualize':
        visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

