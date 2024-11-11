from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

import numpy as np
from tqdm import tqdm
import warnings

# Suppress possible convergence warnings from cross_val_score
warnings.filterwarnings("ignore")

# Define hyperparameter names and their bounds
random_params_name = ['n_estimators', 'min_samples_split', 'min_samples_leaf']
random_params_value = {param: RandomForestClassifier().get_params()[param] for param in random_params_name}
random_bounds = [(10, 200), (2, 10), (1, 4)]

class SkoptRandomForest:
    DEFAULT_MODEL = RandomForestClassifier()
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, score: str = 'accuracy',
                 params_name: list = random_params_name, bounds: list = random_bounds, seed: int = 42):
        """
        Initialize the Bayesian Optimizer for RandomForestClassifier using skopt.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            n_splits (int): Number of cross-validation splits.
            score (str): Scoring metric.
            params_name (list): List of hyperparameter names to optimize.
            bounds (list): List of tuples defining the bounds for each hyperparameter.
            seed (int): Random seed for reproducibility.
        """

        # Validate hyperparameter names
        for param_name in params_name:
            if param_name not in self.DEFAULT_MODEL.get_params().keys():
                raise ValueError(f'Hyperparameter "{param_name}" not found in RandomForestClassifier.')

        np.random.seed(seed)

        self.X_train = X
        self.y_train = y
        self.seed = seed
        self.score = score
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        self.params_name = params_name
        self.bounds = bounds
        self.best_params_ = None
        self.best_score_ = - np.inf

        # Define the search space for skopt
        self.space = []
        for i, param in enumerate(self.params_name):
            low, high = self.bounds[i]
            if isinstance(random_params_value[param], int):
                print('hola')
                self.space.append(Integer(low, high, name=param))
            else:
                self.space.append(Real(low, high, name=param))

    def _evaluate(self, sample: np.array) -> float:
        """
        Evaluate the RandomForestClassifier with given hyperparameters.

        Args:
            params (dict): Dictionary containing hyperparameters.

        Returns:
            float: Mean cross-validated score.
        """ 
        params = {param: (int(round(val)) if isinstance(random_params_value[param], int) else float(val))
                    for param, val in zip(self.params_name, sample)} # Round if integer for each selected param
        
        model = self.DEFAULT_MODEL.set_params(random_state=self.seed, n_jobs=-1, **params)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring=self.score, n_jobs=-1)

        return -scores.mean()
    

    def fit(self, n_iterations: int = 20, n_initial: int = 10, init_type: str = 'random', init_params: dict = None):
        """
        Perform Bayesian Optimization using skopt to find the best hyperparameters.

        Args:
            n_iterations (int): Number of optimization iterations.
            n_initial (int): Number of initial samples.
            init_type (str): Type of initialization ('random', 'default', 'sample').
            init_params (dict, optional): Initial parameters for 'default' or 'sample' types.
        """
        X, y = self._generate_initial(init_type, init_params, n_initial)

        total_calls = n_initial + n_iterations

        self.optimizer = gp_minimize(
            self._evaluate,
            dimensions=self.space,
            acq_func="EI",  
            n_calls=total_calls,
            n_initial_points=0,  
            x0=X,  
            y0=y,  
            random_state=self.seed
        )

        # Correctly retrieve the best score and corresponding hyperparameters
        self.best_score_ = -self.optimizer.fun  # Convert back to positive score
        self.best_params_ = {name: val for name, val in zip(self.params_name, self.optimizer.x)}


    def _generate_initial(self, type: str = 'random', init_params: dict = None, n_initial: int = 10):
        """
        Generate initial set of hyperparameters based on the specified type.

        Args:
            type (str): Type of initialization ('random', 'default', 'sample').
            init_params (dict, optional): Dictionary of hyperparameters for 'default' and 'sample' types.
            n_initial (int): Number of initial points to generate (applicable for 'random' and 'sample').

        Returns:
            X_init (list): List of hyperparameter sets.
            Y_init (list): List of corresponding scores.
        """

        if type == 'default' and init_params:
            sample = [init_params[param] for param in self.params_name]
            return [sample], [self._evaluate(init_params)]

        elif type == 'random':
            X = [
                [
                    np.random.randint(low, high + 1) if isinstance(random_params_value[param], int)
                    else np.random.uniform(low, high)
                    for param, (low, high) in zip(self.params_name, self.bounds)
                ]
                for _ in range(n_initial)
            ]
        else:
            X = [
                [
                    np.clip(
                        np.random.normal(loc=init_params[param], scale=np.random.normal(0, scale=0.05) * (high - low)),
                        low,
                        high
                    ).round() if isinstance(random_params_value[param], int)
                    else np.clip(
                        np.random.normal(loc=init_params[param], scale=np.random.normal(0, scale=0.05) * (high - low)),
                        low,
                        high
                    )
                    for param, (low, high) in zip(self.params_name, self.bounds)
                ]
                for _ in range(n_initial)
            ]
        return X, [self._evaluate(sample) for sample in X]


    def get_best_params(self):
        """
        Get the best hyperparameters found during optimization.

        Returns:
            dict: Best hyperparameters.
        """
        return self.best_params_

    def get_best_score(self):
        """
        Get the best score achieved during optimization.

        Returns:
            float: Best cross-validated score.
        """
        return self.best_score_


def main():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into training and testing sets
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define initial parameters for 'sample' initialization
    initial_params = random_params_value

    # 3. Instantiate and run the optimizer with 'random' initialization
    print("\nRunning optimizer with 'random' initialization:")
    optimizer_random = SkoptRandomForest(
        X=X_train,
        y=y_train,
        n_splits=5,
        score='accuracy',
        params_name=random_params_name,
        bounds=random_bounds,
        seed=42
    )

    optimizer_random.fit(
        n_iterations=20,       
        n_initial=10,          
        init_type='random',  
        init_params=None      
    )

    best_params_random = optimizer_random.get_best_params()
    best_score_random = optimizer_random.get_best_score()

    print("Best Hyperparameters (Random):", best_params_random)
    print("Best Cross-Validated Score (Random):", best_score_random)

    # 4. Instantiate and run the optimizer with 'sample' initialization
    print("\nRunning optimizer with 'sample' initialization:")
    optimizer_sample = SkoptRandomForest(
        X=X_train,
        y=y_train,
        n_splits=5,
        score='accuracy',
        params_name=random_params_name,
        bounds=random_bounds,
        seed=42
    )

    optimizer_sample.fit(
        n_iterations=20,          
        n_initial=10,             
        init_type='sample',        
        init_params=initial_params 
    )

    best_params_sample = optimizer_sample.get_best_params()
    best_score_sample = optimizer_sample.get_best_score()

    print("Best Hyperparameters (Sample):", best_params_sample)
    print("Best Cross-Validated Score (Sample):", best_score_sample)

    # 5. Compare the results
    print("\nComparison of Initialization Methods:")
    print(f"Random Initialization - Best Score: {best_score_random:.4f}")
    print(f"Sample Initialization - Best Score: {best_score_sample:.4f}")




if __name__ == '__main__':
    main()