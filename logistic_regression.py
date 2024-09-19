import numpy as np
import matplotlib.pyplot as plt

def generate_multivariate_data(n):
    
    class_1_mean = [1, 1]
    class_1_cov = [[0.5, 0], [0, 0.5]]

    class_2_mean = [-1, -1]
    class_2_cov = [[0.5, 0], [0, 0.5]]

    # Generate n samples from the multivariate normal distribution for both classes
    x1 = np.random.multivariate_normal(class_1_mean, class_1_cov, n)
    x2 = np.random.multivariate_normal(class_2_mean, class_2_cov, n)

    # Create labels for the two classes
    y1 = np.ones(n)    
    y2 = np.zeros(n)  

    # Combine the data from both classes
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    return x, y

def plot_data(x, y):
    class_1 = x[y == 1]
    class_2 = x[y == 0]

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1', alpha=0.7)
    plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label='Class 2', alpha=0.7)
    
    plt.legend()    
    plt.grid(True)
    

def plot_decision_boundary(x, y, w):

    plt.figure(figsize=(8, 6))
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='Class 1', alpha=0.7)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='red', label='Class 2', alpha=0.7)
    
    # Plot decision boundary where sigmoid(x, w) = 0.5 (i.e., w1*x1 + w2*x2 + w0 = 0)
    x_values = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    y_values = -(w[0] * x_values + w[2]) / w[1]
    
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')
    plt.grid(True)


def predict(x, w):
    prob = sigmoid(x, w)
    return (prob >= 0.5).astype(int)

def sigmoid(x, w): 
    return 1 / (1 + np.exp(-np.dot(x, w)))

def loss(x, y, w):
    prob = sigmoid(x, w)
    loss = -1/x.shape[0] * np.sum(y * np.log(prob) + (1-y)*np.log(1-prob))
    return loss

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

def gradient(x,y,w):
    n = x.shape[0]
    return (1/n) * x.T @ (sigmoid(x,w)-y)

training_iters = 200
learning_rate = 0.05
np.random.seed(1337) 

# Generate our data 
x, y = generate_multivariate_data(100)

# Add a column of ones for our bias
x = np.hstack((x,np.ones((x.shape[0], 1))))

# Define our weight array, two for our dimensions and 1 for bias
w = np.random.normal(size=3)
                   
for i in range(training_iters):
    w -= learning_rate*gradient(x,y,w)
    if i % 10 == 0:
        print("NLL is {}".format(loss(x,y,w)))

y_pred = predict(x, w)
print("Accuracy is {}".format(accuracy(y, y_pred)))






