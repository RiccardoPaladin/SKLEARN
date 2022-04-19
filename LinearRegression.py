import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame= True)
housing = california_housing.data
target = california_housing.target

def most_important_features(housing, target):
    '''
    Funtion that obtaine the coefficients from a linear regression by impoertance
    Input: housing and target datasets
    Output: list of features by importance, in absolute value
    '''
    model = LinearRegression()
    linear_reg = model.fit(housing, target)
    coefficent = list(linear_reg.coef_)
    features = pd.DataFrame()
    features['Features'] = list(housing.columns)
    features['Coefficients'] = coefficent
    features['Coefficients'] = features['Coefficients'].abs()
    features = features.sort_values('Coefficients', ascending = False)
    features_list_by_importance = list(features['Features'])
    print(features_list_by_importance)


if __name__=='__main__':
    most_important_features(housing, target)