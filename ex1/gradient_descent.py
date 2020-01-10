import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(self, alpha=0.01, max_iterations=1500):
        # alpha is the learning rate or size of step
        # to take in the gradient decent
        self._alpha = alpha
        self._max_iterations = max_iterations
        # thetas is the array coefficients for each term
        # the last element is the y-intercept
        self._thetas = None

    def compute_cost(self, xs, ys):
        num_examples, num_features = np.shape(xs)
        self._thetas = np.zeros(num_features)

        # calculate the difference between
        # hypothesis and actual values
        diffs = np.dot(xs, self._thetas) - ys
        cost = np.sum(diffs ** 2) / (2 * num_examples)

        return cost

    def gradient_descent(self, xs, ys):
        num_examples, num_features = np.shape(xs)
        self._thetas = np.zeros(num_features)

        xs_transposed = xs.transpose()
        for i in range(self._max_iterations):
            # calculate the difference between
            # hypothesis and actual values
            diffs = np.dot(xs, self._thetas) - ys
            # calculate gradient for each example
            gradient = np.dot(xs_transposed, diffs) / num_examples
            # update the coefficients
            self._thetas = self._thetas - self._alpha * gradient

        return self._thetas

    def plot_regression_line(self, xs, ys):
        x_primitive = xs[:, 1]
        axes = plt.gca()

        # set the x-axis and y-axis limits
        axes.set_xlim([np.amin(x_primitive), np.amax(x_primitive)])
        axes.set_ylim([np.amin(ys), np.amax(ys)])
		
        # plotting the actual points as scatter plot
        plt.scatter(x_primitive, ys, color="b", marker="x", s=30)
		
        # predicting response vector
        y_predict = np.dot(xs, self._thetas)
		
        # plotting the regression line
        plt.plot(xs, y_predict, color="r")
		
        # putting labels
        plt.xlabel('x')
        plt.ylabel('y')
		
        # function to show plot
        plt.show()


if __name__ == "__main__":
    # load data set
    x = np.loadtxt('ex1data1.txt', delimiter=',', usecols=0)
    y = np.loadtxt('ex1data1.txt', delimiter=',', usecols=1)

    # concatenate an all ones vector as the first column
    x = np.hstack((np.ones((len(y), 1)), x.reshape(97, 1)))

    gd = GradientDescent(alpha=0.01, max_iterations=1500)
    cost = gd.compute_cost(x, y)
    thetas = gd.gradient_descent(x, y)

    gd.plot_regression_line(x, y)
