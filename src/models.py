from src.utils import set_seed

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_MODEL = RandomForestClassifier()
PARAMS_NAME = ['n_estimators', 'min_samples_split', 'min_samples_leaf']
BOUNDS = [(10, 200), (2, 10), (1, 4)]
PARAMS_VALUE = {param: DEFAULT_MODEL.get_params()[param] for param in PARAMS_NAME}

class BoRandomForest:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, score: str = 'accuracy',
                 params_name: list = PARAMS_NAME, bounds: list = BOUNDS, seed: int = 42):
        """
        Initialize the Bayesian Optimizer for RandomForestClassifier.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            n_splits (int): Number of cross-validation splits.
            score (str): Scoring metric.
            params_name (list): List of hyperparameter names to optimize.
            bounds (list): List of tuples defining the bounds for each hyperparameter.
        """

        # Validate hyperparameter names
        for param_name in params_name:
            if param_name not in DEFAULT_MODEL.get_params().keys():
                raise ValueError(f'Hyperparameter "{param_name}" not found in RandomForestClassifier.')

        set_seed(42)

        self.X_train = X
        self.y_train = y
        self.seed = seed
        self.score = score
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        self._params_name = params_name
        self.bounds = torch.tensor(bounds, dtype=torch.float)  
        self.model = RandomForestClassifier
        self.best_params_ = None
        self.best_score_ = - np.inf

    def _evaluate(self, params: dict) -> float:
        """
        Evaluate the RandomForestClassifier with given hyperparameters.

        Args:
            params (dict): Dictionary containing hyperparameters.

        Returns:
            float: Mean cross-validated accuracy.
        """
        model = self.model(random_state=self.seed,
            n_jobs=-1,
            **params,)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring=self.score, n_jobs=-1)
        return scores.mean()

    def _generate_initial(self, type: str = 'random', init_params: dict = None, n_initial: int = 10):
        """
        Generate initial set of hyperparameters based on the specified type.

        Args:
            type (str): Type of initialization ('random', 'default', 'sample').
            init_params (dict, optional): Dictionary of hyperparameters for 'default' and 'sample' types.
            n_initial (int): Number of initial points to generate (applicable for 'random' and 'sample').

        Returns:
            X_init (torch.Tensor): Tensor of shape (n_samples, d)
            Y_init (torch.Tensor): Tensor of shape (n_samples, 1)
        """
        # Validate the 'type' parameter. In the future more validations will be covered.
        if type not in ['random', 'default', 'sample']:
            raise ValueError(f"Unsupported type '{type}'. Supported types are 'random', 'default', 'sample'.")

        Y_init = []
        lower_bounds = self.bounds[:, 0]  
        upper_bounds = self.bounds[:, 1] 
        
        if type == 'random':
            samples = torch.rand(n_initial, len(self._params_name)) * (upper_bounds - lower_bounds) + lower_bounds 
            # TODO
            # Remove duplicates

        elif type == 'default':
            samples = torch.tensor([init_params[param] for param in self._params_name],
                                              dtype=torch.float).unsqueeze(0) 
            
        else:
            center = torch.tensor([init_params[param] for param in self._params_name],
                                      dtype=torch.float) 
            # Define standard deviations as 5% of the bounds range
            std_dev = (upper_bounds - lower_bounds) * np.clip(np.random.normal(loc = 0.0, scale = 0.02), 0, 0.02)
            samples = torch.normal(mean=center.unsqueeze(0).repeat(n_initial, 1),
                                   std=std_dev.unsqueeze(0).repeat(n_initial, 1))  
            # TODO
            # Remove duplicates


        # Within the interval range
        samples = torch.clamp(samples, lower_bounds, upper_bounds)
        # Integer variables
        samples = torch.round(samples).long()

        all_params = samples.tolist()  # List of lists, each inner list is a hyperparameter set

        # Evaluate all initial points.
        for params in all_params:
            param_dict = dict(zip(self._params_name, params))
            Y = self._evaluate(param_dict)
            Y_init.append(Y)


        return samples, torch.tensor(Y_init, dtype=torch.float).unsqueeze(-1)

    def fit(self, n_iterations: int = 20, n_initial: int = 10, init_type: str = 'random', init_params: dict = None):
        """
        Perform Bayesian Optimization using BoTorch to find the best hyperparameters.

        Args:
            n_iterations (int): Number of optimization iterations.
            n_initial (int): Number of initial samples.
            init_type (str): Type of initialization ('random', 'default', 'sample').
            init_params (dict, optional): Initial parameters for 'default' or 'sample' types.
        """
        # Tensors are converted to double as BoTorch recommends
        # Generate initial samples based on init_type
        X_init, Y_init = self._generate_initial(type=init_type, init_params=init_params, n_initial=n_initial)
        X = X_init.double()
        Y = Y_init.squeeze(-1).double()

        # Avoid division by zero
        epsilon = 1e-8
        with tqdm(total=n_iterations, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iterations):
                # Apply scaling as recommended by Botorch
                scaling_factors = self.bounds[:, 1] - self.bounds[:, 0] + epsilon  
                X_normalized = (X - self.bounds[:, 0]) / scaling_factors  

                model = SingleTaskGP(X_normalized, Y.unsqueeze(-1))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                # Log version of EI as it is numerically more stable.
                LEI = LogExpectedImprovement(model, best_f=Y.max().item())

                scaled_bounds = torch.stack([torch.zeros_like(self.bounds[:, 0], dtype=torch.double),
                                            torch.ones_like(self.bounds[:, 1], dtype=torch.double)], dim=1)

                # Optimize acquisition function to find next candidate
                candidate_normalized, _ = optimize_acqf(
                    LEI,
                    bounds=scaled_bounds.t(),  
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )

                # Scale back
                candidate = candidate_normalized * scaling_factors.unsqueeze(0) + self.bounds[:, 0].unsqueeze(0)

                # Here I have a doubt: The optimization works well if I round to integeres?
                candidate = torch.round(candidate).long()
                candidate = torch.clamp(candidate, self.bounds[:, 0].long(), self.bounds[:, 1].long())

                param_dict = dict(zip(self._params_name, candidate.squeeze().tolist()))
                Y_new = self._evaluate(param_dict)

                X = torch.cat([X, candidate.float()], dim=0).double()
                Y = torch.cat([Y, torch.tensor([Y_new], dtype=torch.float)]).double()

                # Update
                if Y_new > self.best_score_:
                    self.best_score_ = Y_new
                    self.best_params_ = param_dict

                pbar.set_postfix({'Best Score': f"{self.best_score_:.4f}"})
                pbar.update(1)








