import numpy as np
import matplotlib.pyplot as plt
import statistics


def compute_cost(x, y, thetas):
    num_examples = np.shape(x)[0]

    # calculate the difference between
    # hypothesis and actual values
    diffs = np.dot(x, thetas) - y
    # calculate cost function
    cost = np.dot(diffs.transpose(), diffs) / (2 * num_examples)

    return cost


def gradient_descent(x, y, thetas, alpha, max_iterations):
    num_examples, num_features = np.shape(x)
    # j_history is for plot convergence graph
    j_history = np.zeros([max_iterations, 1])

    for i in range(max_iterations):
        # create a copy of thetas
        # for simultaneously update
        thetas_previous = thetas

        for j in range(num_features):
            # calculate dJ/d(theta_j)
            derivative = ((np.dot(x, thetas_previous) - y).transpose()) * x[:, j] / num_examples
            # update theta_j
            thetas[j] = thetas_previous[j] - alpha * derivative.sum()

        # save the historic values of cost function
        j_history[i] = compute_cost(x, y, thetas)

    return thetas, j_history


def feature_normalize(x):
    x_normalize = x
    num_examples, num_features = np.shape(x)
    # the average of all the values
    mu = np.zeros(num_features)
    # the standard derivation of all the values
    sigma = np.zeros(num_features)

    for i in range(num_features):
        mu[i] = statistics.mean(x[:, i])

    for i in range(num_features):
        sigma[i] = statistics.stdev(x[:, i])

    for i in range(num_features):
        if sigma[i] != 0:
            for j in range(num_examples):
                # to implement feature scaling and mean normalization
                # by x_normalize = (x-mu)/sigma
                x_normalize[j, i] = (x[j, i] - mu[i]) / sigma[i]

    return x_normalize, mu, sigma


def plot_convergence_graph(x):
    # plot convergence graph
    plt.plot(list(range(1, np.size(x)+1)), x[:, 0], color="r")

    # put labels
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    # show plot
    plt.show()


if __name__ == "__main__":
    # load data set
    # for one variables
    # x = np.loadtxt('ex1data1.txt', delimiter=',', usecols=0)
    # y = np.loadtxt('ex1data1.txt', delimiter=',', usecols=1)

    # for multiple variables
    x = np.loadtxt('ex1data2.txt', delimiter=',', usecols=(0, 1))
    y = np.loadtxt('ex1data2.txt', delimiter=',', usecols=2)

    # alpha is the learning rate or size of step
    # to take in the gradient decent
    alpha = 0.01
    # max value of iterations
    max_iterations = 400

    # for Feature Scaling and Mean Normalization
    # not applicable to one variables linear regression
    normalized = feature_normalize(x)
    x = normalized[0]
    mu = normalized[1]
    sigma = normalized[2]

    # extract number of examples and features
    num_examples, num_features = np.shape(x)

    # concatenate an all ones vector as the first column
    x = np.hstack((np.ones((len(y), 1)), x.reshape(num_examples, num_features)))

    # initialize thetas
    thetas = np.zeros(num_features + 1)

    cost = compute_cost(x, y, thetas)
    gradient_descent = gradient_descent(x, y, thetas, alpha, max_iterations)
    thetas = gradient_descent[0]
    j_history = gradient_descent[1]

    print('cost', cost)
    print('thetas', thetas)

    # plot convergence graph
    plot_convergence_graph(j_history)
