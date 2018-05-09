
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# read
train_df = pd.read_csv('./input/train.csv', index_col=0)
# view
print(train_df.head())

# logarithm of y
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# logarithm of features
numeric_cols = train_df.columns[train_df.dtypes != 'object']
train_df[numeric_cols] = np.log1p(train_df[numeric_cols])

# one hot
train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)
train_df = pd.get_dummies(train_df)
print(train_df.head())
# missing value
train_df.fillna(train_df.mean(), inplace=True)
print(train_df.isnull().sum().sum())
# split

y = train_df.pop('SalePrice')
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=0)


# def rmse_cv_train(model):
#     rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10))
#     return(rmse)
def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5))
    return(rmse)

# ridge
ridge = RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]).fit(X_train, y_train)
alpha = ridge.alpha_
print('Best alpha :', alpha)
print('score:', rmse_cv_train(ridge).mean())
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)


# Plot predictions
plt.scatter(y_train_pred, y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_pred, y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
# plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
plt.show()
