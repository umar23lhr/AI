import numpy as np
import matplotlib.pyplot as plt

# Data
x_train = np.array([1.0, 2.0])  # Size in 1000 sqft
y_train = np.array([300, 500])  # Price in 1000s of dollars

# Task 1: Implement gradient descent algorithm

def compute_cost(x, y, w, b):
    # Cost function implementation (already developed)
    return np.mean((w * x + b - y)**2)

def compute_gradient(x, y, w, b):
    # Compute the gradient for linear regression
    m = len(x)
    dw = (2/m) * np.sum((w * x + b - y) * x)
    db = (2/m) * np.sum(w * x + b - y)
    return dw, db

def gradient_descent(x, y, w, b, alpha, num_iters, cost_function, gradient_function):
    # Perform gradient descent to fit w, b
    J_history = []
    p_history = []

    for _ in range(num_iters):
        dw, db = gradient_function(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        cost = cost_function(x, y, w, b)
        J_history.append(cost)
        p_history.append([w, b])

    return w, b, J_history, p_history

# Task 2: Plotting and Prediction

# Plotting cost versus iterations
def plot_cost_iterations(J_history):
    plt.plot(range(1, len(J_history) + 1), J_history, marker='o')
    plt.title('Cost versus Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

# Predicting housing prices
def predict_prices(x, w, b):
    return w * x + b

# Running gradient descent
w_init = 0
b_init = 0
iterations = 10000
learning_rate = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, iterations, compute_cost, compute_gradient)

# Plotting cost versus iterations
plot_cost_iterations(J_hist)

# Predicting three housing prices
new_sizes = np.array([1.5, 2.5, 3.0])  # New sizes in 1000 sqft
predicted_prices = predict_prices(new_sizes, w_final, b_final)

# Displaying predicted prices
for size, price in zip(new_sizes, predicted_prices):
    print(f"Predicted price for a {size} sqft house: ${price:.2f}")
