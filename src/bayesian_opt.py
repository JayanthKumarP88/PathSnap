import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

class GaussianProcessBayesianOptimizer:
    """
    Principled Bayesian Hyperparameter Optimizer utilizing a Gaussian Process
    surrogate model with Matérn-5/2 kernel and Expected Improvement (EI) acquisition.
    Optimizes group-aware cross-validation score without external heavy dependencies.
    """
    def __init__(self, param_bounds: Dict[str, Tuple[float, float, str]],
                 n_initial_points: int = 5, n_trials: int = 20, random_state: int = 42):
        """
        param_bounds: dict of param_name -> (min_val, max_val, type ['int', 'float', 'log_float'])
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_initial_points = n_initial_points
        self.n_trials = n_trials
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Matern-5/2 kernel with modest noise
        kernel = Matern(nu=2.5, length_scale=np.ones(len(self.param_names))) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True,
                                           n_restarts_optimizer=5, random_state=random_state)

        self.trials_history = []
        self.best_params = None
        self.best_score = -np.inf

    def _sample_random_params(self) -> Dict[str, Any]:
        params = {}
        for name, (low, high, p_type) in self.param_bounds.items():
            if p_type == "int":
                params[name] = int(self.rng.randint(int(low), int(high) + 1))
            elif p_type == "log_float":
                log_val = self.rng.uniform(np.log10(low), np.log10(high))
                params[name] = float(10 ** log_val)
            else: # float
                params[name] = float(self.rng.uniform(low, high))
        return params

    def _params_to_vec(self, params: Dict[str, Any]) -> np.ndarray:
        vec = []
        for name in self.param_names:
            low, high, p_type = self.param_bounds[name]
            val = params[name]
            if p_type == "log_float":
                norm_val = (np.log10(val) - np.log10(low)) / (np.log10(high) - np.log10(low))
            else:
                norm_val = (val - low) / (high - low)
            vec.append(norm_val)
        return np.array(vec, dtype=np.float64)

    def _vec_to_params(self, vec: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, name in enumerate(self.param_names):
            low, high, p_type = self.param_bounds[name]
            norm_val = np.clip(vec[i], 0.0, 1.0)
            if p_type == "int":
                params[name] = int(np.round(low + norm_val * (high - low)))
            elif p_type == "log_float":
                log_val = np.log10(low) + norm_val * (np.log10(high) - np.log10(low))
                params[name] = float(10 ** log_val)
            else:
                params[name] = float(low + norm_val * (high - low))
        return params

    def _expected_improvement(self, X_cand: np.ndarray, xi: float = 0.01) -> np.ndarray:
        mu, sigma = self.gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        improvement = mu - self.best_score - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def optimize(self, objective_func: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        print(f"Starting Bayesian Optimization ({self.n_trials} trials, {self.n_initial_points} random warm-up)...")
        start_time = time.time()

        X_history = []
        y_history = []

        for trial in range(self.n_trials):
            trial_start = time.time()
            if trial < self.n_initial_points:
                params = self._sample_random_params()
                method = "Random Exploration"
            else:
                # Fit GP on observed history
                self.gp.fit(np.vstack(X_history), np.array(y_history))

                # Monte Carlo candidate search for max Expected Improvement
                mc_samples = self.rng.uniform(0.0, 1.0, size=(1000, len(self.param_names)))
                ei_scores = self._expected_improvement(mc_samples)
                best_mc_idx = np.argmax(ei_scores)
                params = self._vec_to_params(mc_samples[best_mc_idx])
                method = "GP Expected Improvement"

            # Evaluate objective
            score = objective_func(params)
            trial_time = time.time() - trial_start

            X_history.append(self._params_to_vec(params))
            y_history.append(score)

            if score > self.best_score:
                self.best_score = score
                self.best_params = params

            record = {
                "trial": trial + 1,
                "strategy": method,
                **params,
                "score": round(score, 5),
                "best_so_far": round(self.best_score, 5),
                "duration_sec": round(trial_time, 2)
            }
            self.trials_history.append(record)
            print(f"Trial {trial+1:02d}/{self.n_trials}: Score = {score:.4f} (Best: {self.best_score:.4f}) | Params: {params}")

        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f}s. Best Score: {self.best_score:.4f}")
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "total_time": total_time,
            "trials_df": pd.DataFrame(self.trials_history)
        }
