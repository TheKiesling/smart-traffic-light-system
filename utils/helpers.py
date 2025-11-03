import os
import pickle
import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd

def save_training_results(results: Dict[str, Any], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_training_results(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_training_metrics(results_dir: str, save_path: str = None):
    progress_file = os.path.join(results_dir, 'progress.csv')
    
    if not os.path.exists(progress_file):
        return
    
    df = pd.read_csv(progress_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métricas de Entrenamiento MAPPO', fontsize=16, fontweight='bold')
    
    if 'episode_reward_mean' in df.columns:
        axes[0, 0].plot(df['training_iteration'], df['episode_reward_mean'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteración')
        axes[0, 0].set_ylabel('Recompensa Promedio')
        axes[0, 0].set_title('Recompensa por Episodio')
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'episode_len_mean' in df.columns:
        axes[0, 1].plot(df['training_iteration'], df['episode_len_mean'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteración')
        axes[0, 1].set_ylabel('Longitud Promedio')
        axes[0, 1].set_title('Longitud del Episodio')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'info/learner/default_policy/learner_stats/policy_loss' in df.columns:
        axes[1, 0].plot(df['training_iteration'], 
                       df['info/learner/default_policy/learner_stats/policy_loss'], 
                       'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteración')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Pérdida de la Política')
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'info/learner/default_policy/learner_stats/vf_loss' in df.columns:
        axes[1, 1].plot(df['training_iteration'], 
                       df['info/learner/default_policy/learner_stats/vf_loss'], 
                       'purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteración')
        axes[1, 1].set_ylabel('Value Function Loss')
        axes[1, 1].set_title('Pérdida de la Función de Valor')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráficas guardadas en: {save_path}")
    
    plt.show()


def print_system_info():   
    if torch.cuda.is_available():
        print(f"  Nombre: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Memoria Asignada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memoria Reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print("="*60 + "\n")


def check_sumo_installation():
    if 'SUMO_HOME' not in os.environ:
        return False
    
    try:
        subprocess.run(['sumo', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return False


def calculate_traffic_metrics(episode_info: Dict[str, Any]) -> Dict[str, float]:
    metrics = {
        'avg_waiting_time': 0.0,
        'avg_speed': 0.0,
        'total_vehicles': 0,
        'throughput': 0.0,
    }
    
    if 'agents' in episode_info:
        waiting_times = [agent_info.get('waiting_time', 0) 
                        for agent_info in episode_info['agents'].values()]
        metrics['avg_waiting_time'] = np.mean(waiting_times) if waiting_times else 0.0
    
    return metrics



