import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ------------------------------------------PART 3(a)----------------------------------------
def compute_cost(X, y, theta):
    m = len(y)  # Number of training examples
    J = 0
    
    # Compute hypothesis/predictions
    h = np.dot(X, theta)
    print("")
    # Compute cost
    J = (1/(2*m)) * np.sum(np.square(h - y))
    
    return J

# ------------------------------------------PART 3(b)----------------------------------------
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)  # Number of training examples
    J_history = np.zeros(num_iters)
    
    for iter in range(num_iters):
        # Compute hypothesis/predictions
        h = np.dot(X, theta)
        
        # Compute gradient
        grad = (1/m) * np.dot(X.T, (h - y))
        
        # Update parameters
        theta = theta - alpha * grad
        
        # Save the cost J in every iteration
        J_history[iter] = compute_cost(X, y, theta)
    
    return theta, J_history

# Generating sample data
X = np.array([[1, 86226, 1956],
              [2, 13248, 1330],
              [3, 60343, 2494],
              [4, 26696, 2494],
              [5, 69414, 2494],
              [6, 49719, 2494],
              [6, 43688, 2494],
              [7, 14470, 2494],
              [8, 21429, 2494],
              [2, 31750, 2494],
              [9, 38203, 2494],
              [8, 110284, 2494],
              [10, 10381, 2494],
              [1, 32378, 2494],
              [8, 38906, 2494],
              [11, 59313, 2494],
              [12, 85672, 2494],
              ])
y = np.array([10.03, 12.83, 16.40, 7.77, 5.15, 7.66, 7.58, 11.60, 6.99, 7.53, 6.43, 5.43, 8.62, 16.78, 10.03, 5.63, 6.67])  # car prices in INR
theta = np.zeros(X.shape[1])  # Initialize model parameters
alpha = 0.01  # Learning rate
num_iters = 1000  # Number of iterations

# Feature scaling (optional)
X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

# Perform gradient descent to obtain the optimal parameters
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)


# Compute the cost function for the learned parameters
cost = compute_cost(X, y, theta)


# Plotting the cost function over iterations
plt.plot(range(num_iters), J_history, 'b')
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.title('Cost function over iterations')

# ------------------------------------------PART 4(a)----------------------------------------

# Check convergence
convergence_threshold = 1e-5  # Set convergence threshold
converged = False

for i in range(1, len(J_history)):
    if abs(J_history[i] - J_history[i-1]) < convergence_threshold:
        converged = True
        print("Algorithm converged after", i, "iterations.")
        break

if not converged:
    print("Algorithm did not converge within the specified threshold.")


# ------------------------------------------PART 4(b)----------------------------------------
# Define a grid of theta0 and theta1 values
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)

# Compute corresponding cost values over the grid
cost_mesh = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta_tmp = np.array([theta0_vals[i], 0, theta1_vals[j]])  # Ensure theta_tmp matches the dimensions of X
        cost_mesh[i, j] = compute_cost(X, y, theta_tmp)

# Plot the cost function surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh, cmap='viridis')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Cost')
plt.title('Cost Function Surface')
plt.show()
# Print the learned parameters
print("Learned parameters (theta):", theta)
print("Cost function computed using learned parameters:", cost)