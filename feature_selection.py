from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import pandas as pd
import numpy as np
from ITMO_FS import su_measure, gini_index, information_gain, spearman_corr, reliefF_measure

def feature_selection(method, X, y, k):
    if method == 'info_gain':
        return info_gain(X, y, k)
    elif method == 'pearson':
        return Pearson(X, y, k)
    elif method == 'spearman':
        return Spearman(X, y, k)
    elif method == 'gini':
        return gini(X, y, k)
    elif method == 'symetrical_uncertainty':
        return symetrical_uncertainty(X, y, k)
    
def Pearson(X, y, k):
    df = X.assign(target=y)
    df = df.apply(lambda x: pd.factorize(x)[0])
    correlation_matrix = df.corr(method='pearson')
    top_k = abs(correlation_matrix['target']).sort_values(ascending=False)[1:k+1]
    return top_k.keys()

def Spearman(X, y, k):
    df = X.assign(target=y)
    df = df.apply(lambda x: pd.factorize(x)[0])
    correlation_matrix = df.corr(method='spearman')
    top_k = abs(correlation_matrix['target']).sort_values(ascending=False)[1:k+1]
    return top_k.keys()

def info_gain(X, y, k):
    X_, y_ = np.array(X), np.array(y)
    scores = information_gain(X_, y_)
    return X.columns[np.argsort(scores)[::-1][:k]]

def gini(X, y, k):
    X_, y_ = np.array(X), np.array(y)
    scores = gini_index(X_, y_)
    return X.columns[np.argsort(scores)[:k]]

def symetrical_uncertainty(X, y, k):
    X_, y_ = np.array(X), np.array(y)
    scores = su_measure(X_, y_)
    return X.columns[np.argsort(scores)[::-1][:k]]



