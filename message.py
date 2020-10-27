import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A fuction to display a given index in the training data set
def drawNum(n):
    if n < 60000:
        image = data[n].reshape((28, 28))
        plt.title('Class Label: {label}'.format(label=label[n]))
        plt.imshow(image, cmap='gray')
        plt.show()

'''
P_c function fo find the probability that the class 'C' appear in the dataset. 
To find the probability, we count the number of occurences of each number 0-9 in the data set and divide by the size of the 
data set. 
'''
def P_c(label):
    prob_c = np.zeros(10)
    for i in range(10):
        for j in range(len(label)):
            if label[j] == i:
                prob_c[i] = prob_c[i] + 1

    return prob_c / len(label)   # array of size 10


# Function to count the total occurances of a given label within the inspected data set (of class labels)
def occurrances(label, n):
    count = 0
    for i in range(len(label)):
        if label[i] == n:
            count = count + 1

    return count    # integer (~6000)

'''
Function to calculate the probability of each pixel value within the given data set. 
This function iterates through all data set rows and averages out the values per label.
Returned is an array of size 10x784.
The function will also draw a plot with a 'heat map' of pixel values for the given digits.
additionally, laplace smoothing is implemented by adding 1 to the numerator 
and 10 (the number of possible classes) to the denominaor.
'''
def Prob_x(data, label):
    px_c = np.zeros((10, 784))
    length = len(label)
    for i in range(10):
        for j in range(length):
            if i == label[j]:
                px_c[i] = px_c[i] + data[j]

        # Laplace smoothing and normalizing
        px_c[i] = (px_c[i] + 1) / (occurrances(label, i) + 10)

        # graphing the heat maps of pixel values
        plt.subplot(2, 5, i + 1)
        image = px_c[i].reshape((28, 28))
        plt.imshow(image)
    plt.show()

    return px_c  # array with dimensions [10, 784]

'''
This function is meant to implement the prediction of test data labels. 
Unfortunately we were not able to implement a 100% Bayes algorithm. 
In this function we multiply the probabilities of a given number, which we obtained from the training data
with the test data.
'''
def predict_fun():
    px_c = Prob_x(data, label)
    p_c = P_c(label)
    p_t = np.zeros(len(testData))
    accuracy = 0

    for i in range(len(testData)):
        pred = 0
        for j in range(10):
            # p_t[i] = np.argmax(np.sum((testData[i]+1) * np.log(px_c[j])) + np.log(p_c[j]))
            predSum = np.sum(testData[i] * px_c[j] * p_c[j])
            if predSum > pred:
                pred = predSum
                p_t[i] = j

        print('Prediction:', p_t[i], '\tActual:', testLabel[i])

        if p_t[i] == testLabel[i]:
            accuracy = accuracy + 1

    print('Accuracy:', accuracy / len(testData)*100)

# '''
#  Prototype for a stable prediction function which should use Bayes theorem to
#  compute the estimated label for the given Test data.
#  This function is currently not functioning due to conflicting array sizes
# '''
# def stable_pred(label, data, testData):
#     new_C = []
#     new_ind = []
#     pxc = Prob_x(label, data)
#     pc = Prob_c(label)
#     for i in range(len(testData)):
#         for j in range(len(Prob_x(label, data))):
#             new_C = (testData[i] * np.log(pxc[j]) + np.log(pc[j]))
#         new_ind.append(np.argmax(new_C))       

#     return new_ind

# Loading Data from given CSV files into usable variables, 
# normalizing by dividing by max value, such that we have values between 0 and 1
data = pd.read_csv('mnist_train.csv', header=None, usecols=[*range(1, 785)]).values / 255
label = pd.read_csv('mnist_train.csv', header=None, usecols=[0]).values
# print(label)
# print(data)

testData = pd.read_csv('mnist_test.csv', header=None, usecols=[*range(1, 785)]).values / 255
testLabel = pd.read_csv('mnist_test.csv', header=None, usecols=[0]).values
# print(testLabel)
# print(testData)


print('Label Probability (0-9):\n', P_c(label))
# drawNum(-1)

predict_fun()
