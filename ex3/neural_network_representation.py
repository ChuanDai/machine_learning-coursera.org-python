import numpy as np
import scipy.io
from scipy.optimize import minimize


def sigmoid(z):
    # sigmoid = 1 / (1 + exp(-z))
    return 1 / (1 + np.exp(-z))


def lr_cost_function(thetas, x, y, lambda_reg, return_grad=False):
    # extract the number of examples
    num_examples = np.shape(x)[0]

    # set hypothesis function
    hypothesis = sigmoid(np.dot(x, thetas))

    # calculate cost
    y_transpose = y.T
    # j1 = -y * log(h(x)), (for y = 1)
    j1 = np.dot(-y_transpose, np.log(hypothesis))
    # j0 = -(1 - y) * log(1 - h(x)), (for y = 0)
    j0 = np.dot(-(1 - y_transpose), np.log(1 - hypothesis))
    # j = (j1 + j0) / num_examples
    j_without_regularization = (j1 + j0) / num_examples
    # j = j_without_regularization + lambda * thetas(j)^2 / 2 * num_examples
    j = j_without_regularization + lambda_reg * np.dot(thetas[1:].T, thetas[1:]) / (2 * num_examples)

    # calculate gradient
    # gradient = dj/d(theta(j)) = (h(x) - y) * x(j) / num_examples
    gradient_without_regularization = np.dot((hypothesis.T - y), x) / num_examples
    # gradient = gradient_without_regularization = (h(x) - y) * x(j) / num_examples, (for j = 0)
    # gradient = gradient_without_regularization + lambda * thetas(j) / num_examples, (for j >= 1)
    gradient = gradient_without_regularization + lambda_reg * np.append(0, thetas[1:]) / num_examples

    if return_grad:
        return j, gradient.flatten()
    else:
        return j


def one_vs_all(x, y, num_labels, lambda_reg):
    # extract the numbers of examples and features
    num_examples, num_features = x.shape

    # initialize thetas
    all_theta = np.zeros((num_labels, num_features + 1))

    # concatenate an all ones vector as the first column
    X = np.column_stack((np.ones((num_examples, 1)), x))

    for c in range(num_labels):
        # initial thetas for current classification
        initial_theta = np.zeros((num_features + 1, 1))

        print("\nTraining {:d} out of {:d} categories...".format(c + 1, num_labels))

        # functions WITH gradient/jac parameter
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        myargs = (X, (y % 10 == c).astype(int), lambda_reg, True)
        theta = minimize(lr_cost_function, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 13},
                         method="Newton-CG", jac=True)

        # for other algorithms and
        # functions WITHOUT gradient/jac parameter
        # refer to
        # https://github.com/arturomp/coursera-machine-learning-in-python/blob/master/mlclass-ex3-004/mlclass-ex3/oneVsAll.py

        # assign row of all_theta corresponding to current classification
        all_theta[c, :] = theta["x"]

    return all_theta


def predict_one_vs_all(all_theta, x):
    # extract the number of examples
    num_examples = x.shape[0]

    # Add ones to the x data matrix
    x = np.column_stack((np.ones((num_examples, 1)), x))

    # calculate predicted vector
    predict = np.argmax(sigmoid(np.dot(x, all_theta.T)), axis=1)

    return predict


if __name__ == "__main__":
    # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
    num_labels = 10

    # load training data
    mat = scipy.io.loadmat('ex3data1.mat')

    X = mat["X"]
    y = mat["y"]

    # crucial step in getting good performance!
    # changes the dimension from (num_examples,1) to (num_examples,)
    y = y.flatten()

    print('Training One-vs-All Logistic Regression...')

    # define lambda
    lambda_reg = 0.1

    all_theta = one_vs_all(X, y, num_labels, lambda_reg)
    predict = predict_one_vs_all(all_theta, X)

    # calculate accuracy
    print('\nTraining Set Accuracy: {:f}'.format((np.mean(predict == y % 10) * 100)))
