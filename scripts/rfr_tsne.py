# Data processing packages
import numpy as np
import pandas as pd
from collections import Counter

# Machine learning packages
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer, FunctionTransformer
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE, SelectPercentile, chi2, mutual_info_regression, SelectFromModel
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import torch

# Visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# Others
import time
from pathlib import Path

X1_pca = torch.load('X1_pca').to_numpy()
X1_ica = torch.load('X1_ica').to_numpy()
X1_tsne = torch.load('X1_tsne').to_numpy()
Y1 = pd.read_csv("Y1.csv", header=None, names=['revenue ']).to_numpy().ravel()
#%%
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X1_pca, Y1, random_state=42, test_size=0.1)
X_train_ica, X_test_ica, _, _ = train_test_split(X1_ica, Y1, random_state=42, test_size=0.1)
X_train_tsne, X_test_tsne, _, _ = train_test_split(X1_tsne, Y1, random_state=42, test_size=0.1)

rfr = RandomForestRegressor(random_state=42)
rfr_param_grid = {
    'n_estimators': [200, 400, 600],
    'bootstrap': [True, False],
     'max_depth': [10],
     'max_features': ['log2'],
     'min_samples_leaf': [2, 4],
     'min_samples_split': [2, 5],
}

rfr_grid_pca = GridSearchCV(rfr, rfr_param_grid, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True,verbose=2)
rfr_grid_ica = GridSearchCV(rfr, rfr_param_grid, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True,verbose=2)
rfr_grid_tsne = GridSearchCV(rfr, rfr_param_grid, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True,verbose=2)
#%%
## Model Selection for Random Forest Regressor
# rfr_grid_pca.fit(X_train_pca, np.log(1 + y_train))
# rfr_grid_ica.fit(X_train_ica, np.log(1 + y_train))
rfr_grid_tsne.fit(X_train_tsne, np.log(1 + y_train))

## Save the grid search results
# torch.save(rfr_grid_pca, "../models/rfr_grid_pca")
# torch.save(rfr_grid_ica, "../models/rfr_grid_ica")
torch.save(rfr_grid_tsne, "../models/rfr_grid_tsne")