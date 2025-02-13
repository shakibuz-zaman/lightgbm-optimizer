#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:30:36 2025

@author: mdshakibuzzaman
"""
import sys
sys.setrecursionlimit(2000)  # Increase the recursion limit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import random
import gc
import os

# Set environment variables to optimize MKL
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Load Dataset
from sklearn.datasets import load_diabetes
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
        sampled_params = {}
        for key, value in self.param_space.items():
            if isinstance(value, tuple):  # Continuous range
                sampled_params[key] = random.uniform(*value)
            elif isinstance(value, range):  # Integer range
                sampled_params[key] = random.choice(value)
            else:  # Categorical
                sampled_params[key] = random.choice(value)
        
        # Ensure parameters are in valid ranges and cast to the correct type
        if "learning_rate" in sampled_params:
            sampled_params["learning_rate"] = max(sampled_params["learning_rate"], 1e-4)  # Clamp to avoid zero
        if "num_leaves" in sampled_params:
            sampled_params["num_leaves"] = int(sampled_params["num_leaves"])  # Ensure integer
        if "max_depth" in sampled_params:
            sampled_params["max_depth"] = int(sampled_params["max_depth"])  # Ensure integer
        if "min_data_in_leaf" in sampled_params:
            sampled_params["min_data_in_leaf"] = int(sampled_params["min_data_in_leaf"])  # Ensure integer

        return sampled_params

    def evaluate(self, params):
        model = self.model(**params, n_jobs=1)  # LightGBM with 1 thread
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)],
                  verbose=False, early_stopping_rounds=10)
        preds = model.predict(self.X_test)
        gc.collect()  # Free memory
        return f1_score(self.y_test, preds)

    def anneal(self):
        current_params = self.random_sample()
        current_score = self.evaluate(current_params)
        self.best_params = current_params
        self.best_score = current_score
        self.scores.append(current_score)  # Record the initial score

        T = self.T
        while T > 1:
            for _ in range(self.max_iter):
                new_params = self.random_sample()
                new_score = self.evaluate(new_params)
                self.scores.append(new_score)  # Record the score for each iteration
                if new_score > current_score or np.exp((new_score - current_score) / T) > random.random():
                    current_params = new_params
                    current_score = new_score
                    if new_score > self.best_score:
                        self.best_params = new_params
                        self.best_score = new_score
                gc.collect()  # Free memory after each iteration
            T *= self.alpha
        return self.best_params, self.best_score, self.scores  # Return scores for plotting

# Hyperparameter Space
param_space = {
    "learning_rate": (0.01, 0.3),
    "num_leaves": (20, 100),         # Integer range
    "max_depth": (3, 10),           # Integer range
    "min_data_in_leaf": (10, 50),   # Integer range
    "subsample": (0.5, 1.0)         # Continuous range
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






# Step 5: Compare with Random Search or Grid Search
from sklearn.model_selection import RandomizedSearchCV

# Track RandomizedSearchCV F1 Scores
f1_scores_random = []  # To store F1 scores for plotting

# Define parameter space for RandomizedSearchCV
param_space_random = {
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "num_leaves": range(20, 100, 10),
    "max_depth": range(3, 10),
    "min_data_in_leaf": range(10, 50, 5),
    "subsample": np.linspace(0.5, 1.0, 5),
}

# Custom scoring function to track F1 scores during RandomizedSearchCV
def custom_scoring(estimator, X, y):
    preds = estimator.predict(X)
    score = f1_score(y, preds)
    f1_scores_random.append(score)  # Append score to the list
    return score

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=LGBMClassifier(),
    param_distributions=param_space_random,
    n_iter=50,  # Number of random samples
    scoring=custom_scoring,
    cv=3,  # 3-fold cross-validation
    verbose=1,
    random_state=42
)

random_search.fit(X_train_sample, y_train_sample)

# Best parameters and scores
best_random_params = random_search.best_params_
best_random_score = random_search.best_score_

print("Best Random Search Params:", best_random_params)
print("Best Random Search F1 Score:", best_random_score)

######################
plt.figure(figsize=(10, 6))


# RandomizedSearchCV
plt.plot(f1_scores_random, marker='x', linestyle='--', label="RandomizedSearchCV", color='r')

plt.title("RandomizedSearchCV F1 Score Progression")
plt.xlabel("Iteration")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.show()
######################

######################
######################
# Plot F1 Score Progression for Both Methods
plt.figure(figsize=(10, 6))

# Simulated Annealing
plt.plot(scores, marker='o', linestyle='-', label="Simulated Annealing", color='b')

# RandomizedSearchCV
plt.plot(f1_scores_random, marker='x', linestyle='--', label="RandomizedSearchCV", color='r')

plt.title("F1 Score Progression: Simulated Annealing vs RandomizedSearchCV")
plt.xlabel("Iteration")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.show()


#######
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score

# Define parameter grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'num_leaves': [20, 40, 60, 80, 100],
    'max_depth': [3, 5, 7, 10],
    'min_data_in_leaf': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.7, 0.9, 1.0]
}

# Define custom scoring function
scorer = make_scorer(f1_score, greater_is_better=True)

# Initialize LightGBM model
lgbm = LGBMClassifier(n_jobs=-1)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,  # 3-fold cross-validation
    verbose=1,
    n_jobs=-1  # Parallel processing
)

# Run GridSearchCV
grid_search.fit(X_train_sample, y_train_sample)

# Best parameters and F1 score
best_grid_params = grid_search.best_params_
best_grid_score = grid_search.best_score_

print("Best Grid Search Parameters:", best_grid_params)
print("Best Grid Search F1 Score:", best_grid_score)
##
results = pd.DataFrame(grid_search.cv_results_)

# Plot the mean F1 score for each hyperparameter combination
plt.figure(figsize=(10, 6))
plt.plot(results['mean_test_score'], marker='o', linestyle='-', color='b', label="Mean F1 Score")
plt.fill_between(
    range(len(results['mean_test_score'])),
    results['mean_test_score'] - results['std_test_score'],  # Lower bound
    results['mean_test_score'] + results['std_test_score'],  # Upper bound
    color='b', alpha=0.2, label="± 1 Std Dev"
)

# Annotate the plot
plt.title("GridSearchCV: F1 Score Across Hyperparameter Combinations")
plt.xlabel("Hyperparameter Combination Index")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.show()
