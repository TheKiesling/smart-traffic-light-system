import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_training(log_dir: str):
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return
    
    csv_path = log_path / "metrics.csv"
    
    if not csv_path.exists():
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"\n📈 ESTADÍSTICAS GENERALES")
    print("-"*60)
    print(f"Total de iteraciones: {len(df)}")
    
    if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
        print(f"\n🎯 RECOMPENSAS")
        print("-"*60)
        print(f"Mejor: {df['reward_mean'].max():.2f}")
        print(f"Peor: {df['reward_mean'].min():.2f}")
        print(f"Promedio: {df['reward_mean'].mean():.2f}")
        print(f"Desviación estándar: {df['reward_mean'].std():.2f}")
        print(f"Última: {df['reward_mean'].iloc[-1]:.2f}")
        
        best_idx = df['reward_mean'].idxmax()
        print(f"\n🏆 Mejor iteración: {df.loc[best_idx, 'iteration']}")
        print(f"   Recompensa: {df.loc[best_idx, 'reward_mean']:.2f}")
        
        rolling_window = min(10, len(df) // 4)
        if rolling_window > 0:
            rolling_mean = df['reward_mean'].rolling(window=rolling_window).mean()
            best_rolling_idx = rolling_mean.idxmax()
            print(f"\n📊 Mejor promedio móvil ({rolling_window} iter): Iteración {df.loc[best_rolling_idx, 'iteration']}")
            print(f"   Recompensa: {rolling_mean.iloc[best_rolling_idx]:.2f}")
        
        improvement = df['reward_mean'].iloc[-1] - df['reward_mean'].iloc[0]
        improvement_pct = (improvement / abs(df['reward_mean'].iloc[0])) * 100
        print(f"\n📈 Mejora total: {improvement:.2f} ({improvement_pct:+.1f}%)")
    
    if 'episode_length' in df.columns and df['episode_length'].notna().any():
        print(f"\n⏱️  DURACIÓN DE EPISODIOS")
        print("-"*60)
        print(f"Promedio: {df['episode_length'].mean():.2f}")
        print(f"Máximo: {df['episode_length'].max():.2f}")
        print(f"Mínimo: {df['episode_length'].min():.2f}")
    
    print(f"\n{'='*60}")
    
    print(f"\n🎨 Creando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Análisis de Entrenamiento - {log_path.name}', 
                 fontsize=16, fontweight='bold')
    
    if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
        ax = axes[0, 0]
        ax.plot(df['iteration'], df['reward_mean'], 'b-', linewidth=1.5, alpha=0.6, label='Original')
        
        rolling_window = min(5, len(df) // 4)
        if rolling_window > 0:
            rolling_mean = df['reward_mean'].rolling(window=rolling_window).mean()
            ax.plot(df['iteration'], rolling_mean, 'r-', linewidth=2.5, label=f'Media móvil ({rolling_window})')
        
        if df['reward_max'].notna().any() and df['reward_min'].notna().any():
            ax.fill_between(df['iteration'], 
                           df['reward_min'], 
                           df['reward_max'], 
                           alpha=0.2, color='blue', label='Rango Min-Max')
        
        ax.axhline(y=df['reward_mean'].max(), color='g', linestyle='--', 
                   alpha=0.5, label=f'Mejor: {df["reward_mean"].max():.2f}')
        
        ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recompensa', fontsize=12, fontweight='bold')
        ax.set_title('Recompensa por Iteración', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    if 'reward_mean' in df.columns and df['reward_mean'].notna().any():
        ax = axes[0, 1]
        improvements = df['reward_mean'].diff()
        ax.bar(df['iteration'], improvements, color=['green' if x > 0 else 'red' for x in improvements], 
               alpha=0.6, width=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cambio en Recompensa', fontsize=12, fontweight='bold')
        ax.set_title('Mejora por Iteración', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    if 'policy_loss' in df.columns and df['policy_loss'].notna().any():
        ax = axes[1, 0]
        ax.plot(df['iteration'], df['policy_loss'], 'r-', linewidth=2, label='Policy Loss')
        if 'vf_loss' in df.columns and df['vf_loss'].notna().any():
            ax.plot(df['iteration'], df['vf_loss'], 'orange', linewidth=2, label='VF Loss')
        ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pérdida (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Pérdidas de Entrenamiento', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    if 'entropy' in df.columns and df['entropy'].notna().any():
        ax = axes[1, 1]
        ax.plot(df['iteration'], df['entropy'], 'purple', linewidth=2)
        ax.fill_between(df['iteration'], 0, df['entropy'], alpha=0.3, color='purple')
        ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax.set_ylabel('Entropía', fontsize=12, fontweight='bold')
        ax.set_title('Entropía de la Política', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Alta exploración')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baja exploración')
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    output_path = log_path / "detailed_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualización guardada en: {output_path}")
    
    plt.show()
    
    print(f"\n📁 Archivos disponibles:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📄 JSON: {log_path / 'metrics.json'}")
    print(f"   📄 Resumen: {log_path / 'summary.txt'}")
    print(f"   📊 Gráfica: {log_path / 'training_metrics.png'}")
    print(f"   📊 Análisis detallado: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Debes proporcionar el directorio de logs")
        print("\nUso:")
        print("    python scripts/analyze_training.py logs/MAPPO_20251101_143000")
        print("\nPara ver todos los entrenamientos disponibles:")
        print("    dir logs")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    analyze_training(log_dir)

