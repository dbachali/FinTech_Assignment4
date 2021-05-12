from sklearn import linear_model
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from pprint import pprint


def linear():
    """Function that creates a linear model for the boston dataset prices and finds the variable
    that is the largest factor"""
    # Set up Boston dataset for linear regression
    # Data is in .data and the price, MDEV, needs to be added from .target
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    # Create array of independent variables
    X = boston.drop('MEDV', axis = 1)
    # Create array of dependent variables (i.e. one variable, price in this case)
    y = boston['MEDV']
    # Fit a linear regression model to the data
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    """ Obtain and print the variable with the largest absolute factor.
    *NOTE: this assumes that all factors are expressed in the same unit of measurement.
    Common practice is to express variables in standardized form
    To standardize, express each value as a deviation from it's mean and divide by the standard deviation.
    This is NOT done here."""
    result = np.where(reg.coef_ == max(reg.coef_, key = abs))
    print('The factor with the most influence is: ', boston.columns[result[0][0]])
    pprint('Looking at the result and what it stands for (ass well as correlation matrix of the data'
           'it doesn\'t seem like NOX should be the most influential factor.  Then again, any major increase'
           'in nitric oxide concentration could make a location uninhabitable, so it is a fair result,'
           'we just normally don\'t see a wide range of NOX values in places where people can live.'
           'Standardizing the factors would account for this and give a more realistic result'
           '(see comment in code).')
    return


def kmeans():
    """Function that utilizes the Elbow heuristic with K Means clustering for the iris dataset prices and
    verifies the correct number of clusters"""
    # Set up Iris dataset (data is in .data)
    iris_dataset = load_iris()
    iris = pd.DataFrame(iris_dataset['data'])

    """Run KMeans for a range of clusters and obtain the distortions for each
    Note: Distortion is the sum of square errors around the centroid of a cluster
    (i.e. a representation of the average distance between the centroids and the data points)."""
    distortions = []
    for i in range(1, 11):
        kModel = KMeans(n_clusters = i)
        kModel.fit(iris)
        distortions.append(kModel.inertia_)

    # Plot the distortions
    plt.plot(range(1, 11), distortions)
    plt.xlabel('# Clusters')
    plt.ylabel('Distortion')
    plt.title('Optimal # Clusters via Elbow Heuristic')
    pprint('Looking at the graph, it can be argued that the size of the distortion tapers off at ~3 clusters.'
           'One might argue that 2 or 4 might be better, but this is too much change in distortion between'
           '2 and 3 clusters and too little change in distortion between 3 and 4 clusters.')
    plt.show()
    return


if __name__ == '__main__':
    # Ask user what they want to do and execute the associated funciton(s)
    command = input('What do you want to do '
                    '(enter "l" for linear regression, "k" for k means clustering, '
                    'or "b" for both)?')
    if command == 'l':
        print("You chose: LINEAR")
        linear()
    elif command == 'k':
        print("You chose: K MEANS CLUSTERING")
        kmeans()
    elif command == 'b':
        print("You chose: BOTH")
        print("LINEAR")
        linear()
        print("K MEANS CLUSTERING")
        kmeans()
    else:
        print('Please enter either l, k, b, or p')
