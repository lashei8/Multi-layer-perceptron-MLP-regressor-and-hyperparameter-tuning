# MLP_Gridsearch.py

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

# Generate synthetic dataset: Sine wave with Gaussian noise
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # Features
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=1000)  # Target with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'hidden_layer_sizes': [(30, 30, 30, 100), (100,), (100, 90), (90,), (90, 100)]
}

# Initialize MLPRegressor with specific settings
mlp_GS = MLPRegressor(activation='relu', random_state=42, max_iter=500)

# Perform GridSearchCV
grid = GridSearchCV(mlp_GS, param_grid, scoring='neg_mean_squared_error', cv=3)
grid.fit(X_train, y_train)

# Get the best model and its predictions
best = grid.best_estimator_
grid_predictions = best.predict(X_test)

# Print the best parameters
print('Best parameters:', grid.best_params_)

# Generate smooth x-values for visualization and predict
x_smooth = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_smooth = best.predict(x_smooth)

# Plot ground truth vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Ground Truth (Test Data)', color='blue', alpha=0.6)
plt.plot(x_smooth, y_smooth, label='Predicted (Smooth Curve)', color='red', alpha=0.8)
plt.title('Ground Truth vs Predicted Values', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("MLPGridsearch_plot.pdf", dpi=300)
plt.show()
