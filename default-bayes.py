from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch

DEFAULT_MODEL = RandomForestClassifier(random_state=42)
PARAMS_NAME = ['n_estimators', 'min_samples_split', 'min_samples_leaf']
BOUNDS = [(10, 200), (1, 10), (1, 4)]
PARAMS_VALUE = {param: DEFAULT_MODEL.get_params()[param] for param in PARAMS_NAME}


class BoRandomForest:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, score: str = 'accuracy',
                 params_name: list = PARAMS_NAME, bounds: list = BOUNDS):
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

        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.score = score
        self._params_name = params_name
        self.bounds = torch.tensor(bounds, dtype=torch.float)  

    def _evaluate(self, params: dict) -> float:
        """
        Evaluate the RandomForestClassifier with given hyperparameters.

        Args:
            params (dict): Dictionary containing hyperparameters.

        Returns:
            float: Mean cross-validated accuracy.
        """
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params,
        )
        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.score, n_jobs=-1)
        return scores.mean()

    def generate_initial(self, type: str = 'random', init_params: dict = None, n_initial: int = 10):
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
            std_dev = (upper_bounds - lower_bounds) * 0.05  #
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

    def fit(self):
        """
        Function where the magic happens.
        Coming soon...
        """
        pass


if __name__ == '__main__':

    # The class is made so that multiple experiments can be made 
    # with a set of prefixed hyperparemeters with variable initia
    # lization values and modes. If no hyperparameters are provided it will use
    # default values of the scikit-learn library.

    # Example of usage
    from sklearn.datasets import load_breast_cancer

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Initialize the optimizer
    optimizer = BoRandomForest(
        X=X,
        y=y,
        n_splits=5,
        score='accuracy',
        params_name=PARAMS_NAME,
        bounds=BOUNDS
    )

    default_params = {param: PARAMS_VALUE[param] for param in PARAMS_NAME}

    # 1. Random Initialization
    X_init_random, Y_init_random = optimizer.generate_initial(type='random', n_initial=2)
    print("Random Initialization:")
    print("X_init_random:", X_init_random)
    print("Y_init_random:", Y_init_random)

    # 2. Default Initialization
    X_init_default, Y_init_default = optimizer.generate_initial(type='default', init_params=default_params, n_initial=0)
    print("\nDefault Initialization:")
    print("X_init_default:", X_init_default)
    print("Y_init_default:", Y_init_default)

    # 3. Sample Initialization
    X_init_sample, Y_init_sample = optimizer.generate_initial(type='sample', init_params=default_params, n_initial=2)
    print("\nSample Initialization:")
    print("X_init_sample:", X_init_sample)
    print("Y_init_sample:", Y_init_sample)


