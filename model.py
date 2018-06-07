
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

X_train = np.load("./output/X_train.npy")
X_test = np.load("./output/X_test.npy")
y_train = np.load("./output/y_train.npy")
y_test = np.load("./output/y_test.npy")


def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


models = [
    LinearRegression(),
    Ridge(),
    BayesianRidge(),
    KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    Lasso(alpha=0.01, max_iter=10000),
    ElasticNet(alpha=0.001, max_iter=10000),
    RandomForestRegressor(),
    SVR(), LinearSVR(),
    XGBRegressor()
]

names = ["LR", "Ridge", "BayR", "KerR", "Lasso", "Ela", "RF", "SVR", "LinSVR", "Xgb"]

for name, model in zip(names, models):
    score = rmse_cv(model, X_train, y_train)
    print("{}: {:.6f}".format(name, score.mean()))

# output:
# LR: 28370409540.815632
# Ridge: 0.124320
# BayR: 0.116093
# KerR: 0.115137
# Lasso: 0.125776
# Ela: 0.115447
# RF: 0.172394
# SVR: 0.124788
# LinSVR: 0.137993
# Xgb: 0.146421

# to predict continuous y, linear model like Ridge, lasso, elastic Net seems stronger
