#1.Create a Basic feedforward neural network for binary classification(use generate  input using NumPy random module)

import numpy as np

# Generate random input data
num_samples = 1000
X = np.random.rand(num_samples, 2)  # Two features
Y = np.random.randint(2, size=num_samples)  # Binary labels (0 or 1)

from keras.models import Sequential
from keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=2))  # Input layer
model.add(Dense(3, activation='relu'))  # Hidden layer
model.add(Dense(1,units=1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=10,batch_size=32)

#2.Create a Neural Network model by your own name with a random input function of shape 300*4 for Binary classification

# Step 1: Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Step 2: Generate some random data
np.random.seed(0)
X = np.random.rand(300, 4)  # Features (4 input neurons)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target variable

## Step 3: Create a Sequential model
my_model = Sequential()

# Step 4: Add layers to the model
my_model.add(Dense(5, input_dim=4, activation='relu'))  # Hidden layer with 5 neurons
my_model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (binary classification)

# Step 5: Compile the model
my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train the model
my_model.fit(X, y, epochs=15, batch_size=20)

# Step 7: Evaluate the model
loss, accuracy = my_model.evaluate(X, y)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy*100:.2f}%')

#3.Write a Python code using Keras to perform linear regression on randomly generated data.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#Generating random data
np.random.seed(0)
X = np.random.rand(200, 1)
y = 4 * X + 3 + np.random.randn(200, 1) * 0.1  # Adding some noise

#Creating a neural network model
model = Sequential()
model.add(Dense(1, input_dim=1))

# Compiling the model
model.compile(optimizer='sgd', loss='mse')

# Training the model
history = model.fit(X, y, epochs=200, verbose=0)

# Plotting the data and the regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Keras')
plt.show()
print(f'Loss: {loss:.4f}')

