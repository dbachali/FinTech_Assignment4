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
    print("linear")

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
