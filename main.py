import sys
import random
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.datasets import load_diabetes
from lightgbm import LGBMClassifier

# Set recursion limit and optimize MKL environment variables
sys.setrecursionlimit(2000)
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Load Dataset
data = load_diabetes(as_frame=True)
X = data.data
y = (data.target > data.target.median()).astype(int)  # Binary classification

# Sample a smaller dataset for tuning
X_train_sample, _, y_train_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)

# Define Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, model, param_space, X_train, y_train, X_test, y_test, T=200, alpha=0.85, max_iter=25):
        self.model = model
        self.param_space = param_space
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.T = T
        self.alpha = alpha
        self.max_iter = max_iter
        self.best_params = None
        self.best_score = -np.inf
        self.scores = []  # Track F1 scores over iterations

    def random_sample(self):
        """Sample random parameters from the parameter space."""
        sampled_params = {}
        for key, value in self.param_space.items():
            if isinstance(value, tuple):  # Continuous range
                sampled_params[key] = random.uniform(*value)
            elif isinstance(value, range):  # Integer range
                sampled_params[key] = random.choice(value)
            else:  # Categorical
                sampled_params[key] = random.choice(value)

        # Ensure parameters are in valid ranges and cast to correct types
        sampled_params = {
            "learning_rate": max(sampled_params.get("learning_rate", 0), 1e-4),
            "num_leaves": int(sampled_params.get("num_leaves", 20)),
            "max_depth": int(sampled_params.get("max_depth", 3)),
            "min_data_in_leaf": int(sampled_params.get("min_data_in_leaf", 10)),
            "subsample": sampled_params.get("subsample", 1.0)
        }
        return sampled_params

    def evaluate(self, params):
        """Train and evaluate the model with given parameters."""
        model = self.model(**params, n_jobs=1)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)],
                  verbose=False, early_stopping_rounds=10)
        preds = model.predict(self.X_test)
        gc.collect()
        return f1_score(self.y_test, preds)

    def anneal(self):
        """Perform simulated annealing to optimize hyperparameters."""
        current_params = self.random_sample()
        current_score = self.evaluate(current_params)
        self.best_params = current_params
        self.best_score = current_score
        self.scores.append(current_score)

        T = self.T
        while T > 1:
            for _ in range(self.max_iter):
                new_params = self.random_sample()
                new_score = self.evaluate(new_params)
                self.scores.append(new_score)
                if new_score > current_score or np.exp((new_score - current_score) / T) > random.random():
                    current_params = new_params
                    current_score = new_score
                    if new_score > self.best_score:
                        self.best_params = new_params
                        self.best_score = new_score
                gc.collect()
            T *= self.alpha
        return self.best_params, self.best_score, self.scores

# Define Hyperparameter Space
param_space = {
    "learning_rate": (0.01, 0.3),
    "num_leaves": range(20, 100),
    "max_depth": range(3, 10),
    "min_data_in_leaf": range(10, 50),
    "subsample": (0.5, 1.0)
}

# Run Simulated Annealing
sa = SimulatedAnnealing(LGBMClassifier, param_space, X_train_sample, y_train_sample, X, y)
best_params, best_score, scores = sa.anneal()

print("Best Params:", best_params)
print("Best Score:", best_score)

# Plot F1 Score Progression
plt.figure(figsize=(10, 6))
plt.plot(scores, marker='o', linestyle='-', color='b')
plt.title("Simulated Annealing F1 Score Progression")
plt.xlabel("Iteration")
plt.ylabel("F1 Score")
plt.grid()
plt.show()

# Randomized Search for Comparison
param_space_random = {
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "num_leaves": range(20, 100, 10),
    "max_depth": range(3, 10),
    "min_data_in_leaf": range(10, 50, 5),
    "subsample": np.linspace(0.5, 1.0, 5)
}

f1_scores_random = []

def custom_scoring(estimator, X, y):
    preds = estimator.predict(X)
    score = f1_score(y, preds)
    f1_scores_random.append(score)
    return score

random_search = RandomizedSearchCV(
    estimator=LGBMClassifier(),
    param_distributions=param_space_random,
    n_iter=50,
    scoring=custom_scoring,
    cv=3,
    verbose=1,
    random_state=42
)
random_search.fit(X_train_sample, y_train_sample)

print("Best Random Search Params:", random_search.best_params_)
print("Best Random Search F1 Score:", random_search.best_score_)

plt.figure(figsize=(10, 6))
plt.plot(f1_scores_random, marker='x', linestyle='--', color='r', label="RandomizedSearchCV")
plt.title("RandomizedSearchCV F1 Score Progression")
plt.xlabel("Iteration")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.show()

# Grid Search for Comparison
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'num_leaves': [20, 40, 60, 80, 100],
    'max_depth': [3, 5, 7, 10],
    'min_data_in_leaf': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.7, 0.9, 1.0]
}

scorer = make_scorer(f1_score, greater_is_better=True)
lgbm = LGBMClassifier(n_jobs=-1)

grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_sample, y_train_sample)

print("Best Grid Search Parameters:", grid_search.best_params_)
print("Best Grid Search F1 Score:", grid_search.best_score_)

results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['mean_test_score'], marker='o', linestyle='-', color='b', label="Mean F1 Score")
plt.fill_between(
    range(len(results['mean_test_score'])),
    results['mean_test_score'] - results['std_test_score'],
    results['mean_test_score'] + results['std_test_score'],
    color='b', alpha=0.2, label="\u00b1 1 Std Dev"
)
plt.title("GridSearchCV: F1 Score Across Hyperparameter Combinations")
plt.xlabel("Hyperparameter Combination Index")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.show()
