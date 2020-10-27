# Importing usefull modules (Numpy for Math and Pandas to handle CSV file)
import numpy as np
import pandas as pd

l_r = 0.001         # Learning Rate
m_it = 100000       # Maximum iterations

# Preprocessing Input data
readData = pd.read_csv('D3.csv', header=None)
data = readData.values

# Extracting the X and Y values from the data matrix
X = data[:,:-1]
X = np.insert(X, 0, 1, axis=1)
Y = data[:, -1].reshape(X.shape[0],1)

# Breaking the X values into individual 100 by 1 matrices 
X0 = X[:, 0].reshape(100,1)
X1 = X[:, 1].reshape(100,1)
X2 = X[:, 2].reshape(100,1)
X3 = X[:, 3].reshape(100,1)


# Gradient Descent function implementation
def gradDescent(theta, X, Y, l_r, m_it):
    for i in range(m_it):
        # Hypothesis equation
        pred = (theta[0] * X0) + (theta[1] * X1) + (theta[2] * X2) + (theta[3] * X3)
        var = pred - Y

        loss = sum(np.square(var)) / Y.shape[0] / 2

        if loss < l_r:
            break

        
        theta[0] = theta[0] - (l_r / Y.shape[0] * np.sum(var * X0))
        theta[1] = theta[1] - (l_r / Y.shape[0] * np.sum(var * X1))
        theta[2] = theta[2] - (l_r / Y.shape[0] * np.sum(var * X2))
        theta[3] = theta[3] - (l_r / Y.shape[0] * np.sum(var * X3))

    return theta

# Creating a matrix of ones for storage of theta values
theta = np.ones((4,1))
# running gradient descent and storing values in theta vector
theta = gradDescent(theta, X, Y, l_r, m_it)

print("Theta Values: \n")
print(theta)

inputVal = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
pred1 = (theta[0] + (theta[1]*inputVal[0,0]) + (theta[2] * inputVal[0,1]) + (theta[3] * inputVal[0,2]))
pred2 = (theta[0] + (theta[1]*inputVal[1,0]) + (theta[2] * inputVal[1,1]) + (theta[3] * inputVal[1,2]))
pred3 = (theta[0] + (theta[1]*inputVal[2,0]) + (theta[2] * inputVal[2,1]) + (theta[3] * inputVal[2,2]))

print('\nTest Values:', inputVal[0,:])
print('Prediction:', pred1)

print('\nTest Values:', inputVal[1,:])
print('Prediction:', pred2)

print('\nTest Values:', inputVal[2,:])
print('Prediction:', pred3)