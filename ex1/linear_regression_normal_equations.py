import numpy as np
import matplotlib.pyplot as plt


def normal_equations(x, y):
    # extract numbers for examples and features
    if len(x.shape) == 1:
        num_examples = np.shape(x)[0]
        num_features = 1
    else:
        num_examples, num_features = np.shape(x)

    # concatenate an all ones vector as the first column
    x = np.hstack((np.ones((len(y), 1)), x.reshape(num_examples, num_features)))

    # x's transposition
    x_transpose = x.transpose()

    # calculate thetas by normal equations
    thetas = np.dot(np.linalg.inv(np.dot(x_transpose, x)), np.dot(x_transpose, y))

    return thetas


def plot_regression_line(x, y, b):
    # plot the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predict response vector
    y_predict = b[0] + b[1] * x
    # plot the regression line
    plt.plot(x, y_predict, color="g")

    # put labels
    plt.xlabel('x')
    plt.ylabel('y')

    # show plot
    plt.show()


if __name__ == "__main__":
    # load data set
    # for one variable
    # x = np.loadtxt('ex1data1.txt', delimiter=',', usecols=0)
    # y = np.loadtxt('ex1data1.txt', delimiter=',', usecols=1)

    # for multiple variables
    x = np.loadtxt('ex1data2.txt', delimiter=',', usecols=(0, 1))
    y = np.loadtxt('ex1data2.txt', delimiter=',', usecols=2)

    thetas = normal_equations(x, y)
    print(thetas)

    # only available for one variable regression
    # plot_regression_line(x, y, thetas)
