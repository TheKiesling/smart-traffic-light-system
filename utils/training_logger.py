import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"training_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        
        self.csv_path = os.path.join(self.experiment_dir, "metrics.csv")
        self.json_path = os.path.join(self.experiment_dir, "metrics.json")
        self.summary_path = os.path.join(self.experiment_dir, "summary.txt")
    
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        log_entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
        }
        
        reward_keys = [
            'episode_reward_mean',
            'env_runners/episode_reward_mean',
            'sampler_results/episode_reward_mean',
            'evaluation/episode_reward_mean'
        ]
        
        for key in reward_keys:
            if key in metrics:
                log_entry['reward_mean'] = metrics[key]
                break
        
        if 'reward_mean' not in log_entry:
            log_entry['reward_mean'] = None
        
        reward_max_keys = [
            'episode_reward_max',
            'env_runners/episode_reward_max',
            'sampler_results/episode_reward_max'
        ]
        
        for key in reward_max_keys:
            if key in metrics:
                log_entry['reward_max'] = metrics[key]
                break
        
        if 'reward_max' not in log_entry:
            log_entry['reward_max'] = None
        
        reward_min_keys = [
            'episode_reward_min',
            'env_runners/episode_reward_min',
            'sampler_results/episode_reward_min'
        ]
        
        for key in reward_min_keys:
            if key in metrics:
                log_entry['reward_min'] = metrics[key]
                break
        
        if 'reward_min' not in log_entry:
            log_entry['reward_min'] = None
        
        episode_len_keys = [
            'episode_len_mean',
            'env_runners/episode_len_mean',
            'sampler_results/episode_len_mean'
        ]
        
        for key in episode_len_keys:
            if key in metrics:
                log_entry['episode_length'] = metrics[key]
                break
        
        if 'episode_length' not in log_entry:
            log_entry['episode_length'] = None
        
        log_entry['learning_rate'] = metrics.get('info/learner/default_policy/cur_lr', 
                                                  metrics.get('info/learner/shared_policy/cur_lr', None))
        
        log_entry['policy_loss'] = metrics.get('info/learner/default_policy/learner_stats/policy_loss',
                                               metrics.get('info/learner/shared_policy/learner_stats/policy_loss', None))
        
        log_entry['vf_loss'] = metrics.get('info/learner/default_policy/learner_stats/vf_loss',
                                           metrics.get('info/learner/shared_policy/learner_stats/vf_loss', None))
        
        log_entry['entropy'] = metrics.get('info/learner/default_policy/learner_stats/entropy',
                                           metrics.get('info/learner/shared_policy/learner_stats/entropy', None))
        
        self.metrics_history.append(log_entry)
        
        self._save_metrics()
        
        if log_entry['reward_mean'] is not None:
            print(f"   📈 Iteración {iteration}: Recompensa = {log_entry['reward_mean']:.2f}")
    
    def _save_metrics(self):
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.csv_path, index=False)
        
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_summary(self):
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"RESUMEN DE ENTRENAMIENTO - {self.experiment_name}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de iteraciones: {len(self.metrics_history)}\n\n")
            
            if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
                f.write("RECOMPENSAS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mejor recompensa: {df['reward_mean'].max():.2f}\n")
                f.write(f"  Peor recompensa: {df['reward_mean'].min():.2f}\n")
                f.write(f"  Recompensa promedio: {df['reward_mean'].mean():.2f}\n")
                f.write(f"  Desviación estándar: {df['reward_mean'].std():.2f}\n")
                f.write(f"  Recompensa final: {df['reward_mean'].iloc[-1]:.2f}\n\n")
            
            if 'episode_length' in df.columns and df['episode_length'].notna().any():
                f.write("LONGITUD DE EPISODIOS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Promedio: {df['episode_length'].mean():.2f}\n")
                f.write(f"  Máximo: {df['episode_length'].max():.2f}\n")
                f.write(f"  Mínimo: {df['episode_length'].min():.2f}\n\n")
            
            if 'policy_loss' in df.columns and df['policy_loss'].notna().any():
                f.write("PÉRDIDAS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Policy Loss final: {df['policy_loss'].iloc[-1]:.6f}\n")
                f.write(f"  VF Loss final: {df['vf_loss'].iloc[-1]:.6f}\n")
                f.write(f"  Entropy final: {df['entropy'].iloc[-1]:.6f}\n\n")
            
            f.write("="*60 + "\n")
        
        print(f"\n✅ Resumen guardado en: {self.summary_path}")
    
    def plot_metrics(self, save_path: str = None):
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Métricas de Entrenamiento - {self.experiment_name}', 
                     fontsize=16, fontweight='bold')
        
        if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
            ax = axes[0, 0]
            ax.plot(df['iteration'], df['reward_mean'], 'b-', linewidth=2, label='Media')
            
            if df['reward_max'].notna().any() and df['reward_min'].notna().any():
                ax.fill_between(df['iteration'], 
                               df['reward_min'], 
                               df['reward_max'], 
                               alpha=0.3, label='Min-Max')
            
            ax.set_xlabel('Iteración', fontsize=12)
            ax.set_ylabel('Recompensa', fontsize=12)
            ax.set_title('Recompensa por Iteración', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        if 'episode_length' in df.columns and df['episode_length'].notna().any():
            ax = axes[0, 1]
            ax.plot(df['iteration'], df['episode_length'], 'g-', linewidth=2)
            ax.set_xlabel('Iteración', fontsize=12)
            ax.set_ylabel('Longitud', fontsize=12)
            ax.set_title('Longitud de Episodio', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        if 'policy_loss' in df.columns and df['policy_loss'].notna().any():
            ax = axes[1, 0]
            ax.plot(df['iteration'], df['policy_loss'], 'r-', linewidth=2, label='Policy Loss')
            if df['vf_loss'].notna().any():
                ax.plot(df['iteration'], df['vf_loss'], 'orange', linewidth=2, label='VF Loss')
            ax.set_xlabel('Iteración', fontsize=12)
            ax.set_ylabel('Pérdida', fontsize=12)
            ax.set_title('Pérdidas de Entrenamiento', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_yscale('log')
        
        if 'entropy' in df.columns and df['entropy'].notna().any():
            ax = axes[1, 1]
            ax.plot(df['iteration'], df['entropy'], 'purple', linewidth=2)
            ax.set_xlabel('Iteración', fontsize=12)
            ax.set_ylabel('Entropía', fontsize=12)
            ax.set_title('Entropía de la Política', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, "training_metrics.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráficas guardadas en: {save_path}")
        
        plt.close()
    
    def get_best_iteration(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return None
        
        df = pd.DataFrame(self.metrics_history)
        
        if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
            best_idx = df['reward_mean'].idxmax()
            best_iteration = df.loc[best_idx].to_dict()
            return best_iteration
        
        return None

