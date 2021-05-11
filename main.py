from sklearn import linear_model
from sklearn.datasets import load_boston
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


if __name__ == '__main__':
