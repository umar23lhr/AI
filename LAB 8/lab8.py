import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset
X = np.load('X.npy')
y = np.load('y.npy')

# Task 1: Visualizing the Data
def visualize_data(X, y, num_samples=64):
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    images = X[indices]
    labels = y[indices].flatten()
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(20, 20), cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {labels[i]}")
    plt.show()

visualize_data(X, y)

# Task 2: Build a Neural Network using TensorFlow and Keras
model = Sequential([
    Dense(25, activation='relu', input_shape=(400,)),
    Dense(15, activation='relu'),
    Dense(10)  # No activation for the output layer
])

# Task 3: Examine the weights in the layers
for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Layer Weights Shape:", weights.shape)
    print("Layer Biases Shape:", biases.shape)

# Task 4: Define a loss function and an optimizer
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=Adam(), loss=loss_fn, metrics=['accuracy'])

# Task 5: Analyze the loss(cost) of training and algorithm convergence by plotting the loss
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Task 6: Use Keras predict function to the image of two X[1015]
prediction = model.predict(X[1015:1016])
print("Prediction for X[1015]:", prediction)
