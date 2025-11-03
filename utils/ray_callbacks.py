from ray.tune import Callback
from typing import Dict, Any
from utils.training_logger import TrainingLogger


class MetricsLoggerCallback(Callback):
    
    def __init__(self, logger: TrainingLogger):
        super().__init__()
        self.logger = logger
        self.iteration_count = 0
    
    def on_trial_result(self, iteration: int, trials, trial, result: Dict[str, Any], **info):
        training_iteration = result.get('training_iteration', iteration)
        
        self.logger.log_iteration(training_iteration, result)
        self.iteration_count += 1
    
    def on_trial_complete(self, iteration: int, trials, trial, **info):       
        self.logger.save_summary()
        
        try:
            self.logger.plot_metrics()
        except Exception as e:
            print(f"No se pudieron generar gráficas: {e}")
        
        best_iteration = self.logger.get_best_iteration()
        if best_iteration:
            print(f"\n🏆 Mejor iteración: {best_iteration['iteration']}")
            if best_iteration.get('reward_mean') is not None:
                print(f"   Recompensa: {best_iteration['reward_mean']:.2f}")

