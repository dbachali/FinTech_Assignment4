from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up Boston dataset for linear regression
# Data is in .data and the price, MDEV, needs to be added from .target
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target


def correlation():
    return boston.corr()

def linear():
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
    print("The factor with the most influence is: ", boston.columns[result[0][0]])

    """Looking at the result and what it stands for (ass well as correlation matrix of the data),
     it doesn't seem like NOX should be the most influential factor.  Then again, any major increase
     in nitric oxide concentration could make a location unihabitable, so it is a fair result,
     we just normally don't see a wide range of NOX values in places where people can live.
     As in the comment above, standardizing the factors would account for this and give a more realistic result."""


def kmeans():
    print("kmeans")

if __name__ == '__main__':
    command = input('What do you want to do (enter "l" for linear regression, "k" for k-means clustering, "b" for both?')
    if command == "l":
        linear()
    elif command == "k":
        kmeans()
    elif command == "b":
        linear()
        kmeans()
    else:
        print('Please enter either l, k, or b')
