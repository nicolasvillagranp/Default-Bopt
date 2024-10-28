# Data libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

#BoRandom Model
from src.models import BoRandomForest, PARAMS_VALUE

def test_dataset(X: np.ndarray, y: np.array, n_iterations: int = 5, n_initial: int = 10, seed: int = 42):
    
    # 2. Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 3. Apply standarization to the data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 3. Define initialization parameters for 'sample' type
    init_params = PARAMS_VALUE  # Using default hyperparameters as the center for sampling


    optimizer_random = BoRandomForest(X_train, y_train)

    optimizer_random.fit(
        n_iterations=n_iterations,
        n_initial=n_initial,
        init_type='random',
    )



    print("Best Hyperparameters (Random Init):", optimizer_random.best_params_)
    print(f"Best {optimizer_random.score}: {optimizer_random.best_score_}\n")


    
    optimizer_sample = BoRandomForest(X_train, y_train)
    optimizer_sample.fit(
        n_iterations=n_iterations,
        n_initial=n_initial,
        init_type='sample',
        init_params=init_params  # Providing initial parameters for 'sample' type
    )


    print("Best Hyperparameters (Sample Init):", optimizer_sample.best_params_)
    print(f"Best {optimizer_sample.score}: {optimizer_sample.best_score_}\n")


    # Check behaviour in test set. 

    # best_model_random = RandomForestClassifier(**optimizer_random.best_params_, random_state=seed, n_jobs=-1)
    # best_model_random.fit(X_train, y_train)
    # test_score_random = best_model_random.score(X_test, y_test)
    # print(f"Test Score (Random Init Best Model): {test_score_random}")

    # best_model_sample = RandomForestClassifier(**optimizer_sample.best_params_, random_state=seed, n_jobs=-1)
    # best_model_sample.fit(X_train, y_train)
    # test_score_sample = best_model_sample.score(X_test, y_test)
    # print(f"Test Score (Sample Init Best Model): {test_score_sample}")

    return optimizer_sample.best_score_, optimizer_random.best_score_