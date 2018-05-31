import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, RobustScaler
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

os.chdir('F:\python\machine learning\HousePrice')
pd.set_option('display.width', 500, 'display.max_rows', 500)  # 'precision', 2
train = pd.read_csv('./input/train.csv')


# data explore
print(train.head())
# Id column is useless
train.drop('Id', axis=1, inplace=True)
print(train.columns, train.dtypes)
print(train.describe())


# correlation analysis
corr_df = train.select_dtypes(exclude=['object']).corr()

# correlation with y
corr_with_y = corr_df['SalePrice'].to_dict()
del corr_with_y['SalePrice']
print(' the correlation of numerical features and SalesPrice:\n')
for ele in sorted(corr_with_y.items(), key=lambda x: -abs(x[1])):
    print('{0}: {1}'.format(*ele))

# mutual correlation between features
high_corr_list = []
feats = []
for i, corr in corr_df[(corr_df >= 0.6) & (corr_df != 1.0)].to_dict().items():
    for k, v in corr.items():
        if pd.isnull(v):  # 此处有坑：如果用 if v == np.nan来判断，如果v是nan也会返回False，因为nan是float的子类,
            # 正确的判断方法：np.isnan(), pd.isnull(), import math math.isnan()
            continue
        elif i != 'SalePrice' and k != 'SalePrice':
            # print(i, k, v)
            if {i, k} not in feats:  # type({i, k}): set
                feats.append({i, k})
                high_corr_list.append((i, k, v))
high_corr_df = pd.DataFrame(high_corr_list, columns=['feat1', 'feat2', 'feat1_vs_feat2'])
del high_corr_list
gc.collect()

feat1_vs_y, feat2_vs_y = [], []
for f in high_corr_df['feat1']:
    feat1_vs_y.append(corr_df.loc[f, 'SalePrice'])
for f in high_corr_df['feat2']:
    feat2_vs_y.append(corr_df.loc[f, 'SalePrice'])
high_corr_df['feat1_vs_y'] = feat1_vs_y
high_corr_df['feat2_vs_y'] = feat2_vs_y
del feat1_vs_y, feat2_vs_y
gc.collect()
delete_feats = []
for i in high_corr_df.index:
    if high_corr_df.loc[i, 'feat1_vs_y'] >= high_corr_df.loc[i, 'feat2_vs_y']:
        delete_feats.append(high_corr_df.loc[i, 'feat2'])
    else:
        delete_feats.append(high_corr_df.loc[i, 'feat1'])
high_corr_df['delete_feat'] = delete_feats
high_corr_df.sort_values(axis=0, ascending=False, by=['feat1_vs_feat2', 'feat1_vs_y', 'feat2_vs_y'], inplace=True)
print(high_corr_df)

# drop feats
orig_feat_nums = len(train.columns)
for f in set(delete_feats):
    train.drop(f, inplace=True, axis=1)
    print('> %s deleted from train' % f)
delta = (orig_feat_nums - len(train.columns))
print('>> deleted %d features' % delta)
del orig_feat_nums, delta
gc.collect()

# visual
# heat map
corr_df = train.select_dtypes(exclude=['object']).corr()
k = 10  # number of variables for heatmap
cols = corr_df.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)  # control font size of x and y label
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
# plt.show()
# pair plot
k = 4
cols = corr_df.nlargest(k, 'SalePrice')['SalePrice'].index
sns.set()
sns.pairplot(train[cols])
# plt.show()


# outliers of main feats
print(corr_df.nlargest(8, 'SalePrice')['SalePrice'].index)

# 1. OverallQual
print(train.dtypes['OverallQual'])  # Overall material and finish quality
print(train['OverallQual'].describe())
train['OverallQual'] = train['OverallQual'].astype('category')  # change type to category
# plot
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(train.OverallQual, train.SalePrice)
fig.axis(ymin=0, ymax=800000)
# plt.show()
# # the plot reveals a strong correlation and seem no outliers

# 2. GrLivArea
# temp_df = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
# temp_df.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# # an alternative to plot: no new dataframe was created, recommended
plt.figure(figsize=(8, 6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.ylim(0, 800000)
# plt.show()
# the plot reveals two outliers, drop them
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace=True)
print('>> dropped two outliers of GrLivArea')

# 3. TotalBsmtSF
plt.figure(figsize=(8, 6))
plt.scatter(x=train.TotalBsmtSF, y=train.SalePrice)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.ylim(0, 800000)
# plt.show()
# the outliers of TotalBsmtSF already deleted when drop outliers of GrLivArea

# 4. YearBuilt
plt.figure(figsize=(12, 8))
sns.boxplot(train.YearBuilt, train.SalePrice)
plt.xticks(rotation=90)
# plt.show()
# non remarkable tendency, let it be; any other way???

# 5. YearRemodAdd
plt.figure(figsize=(12, 8))
sns.boxplot(train.YearRemodAdd, train.SalePrice)
plt.xticks(rotation=90)
# plt.show()
# newly add houses seem have higher price

# 6. MasVnrArea: Masonry veneer area...
plt.figure(figsize=(8, 6))
plt.scatter(x=train.MasVnrArea, y=train.SalePrice)
plt.xlabel('MasVnrArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.ylim(0, 800000)
# plt.show()
# it is wired that this feature could affect house price,
# after all, nobody will take this into consideration when buying a house
print(corr_df.loc['MasVnrArea', 'SalePrice'])  # 0.477493047096
# it turns out that it does have effects on price..., so keep it

# 7. Fireplaces: Number of fireplaces
plt.figure(figsize=(8, 6))
plt.scatter(x=train.Fireplaces, y=train.SalePrice)
plt.xlabel('Fireplaces', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.ylim(0, 800000)
# plt.show()
# no evident trend, can not spot outliers...


# missing data
# calculate total missing value and missing percent
missing_df = train.isnull().sum()
missing_df = missing_df[missing_df > 0]  # remove 0 missing data columns
pct_df = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
# pct_df = pct_df.applymap(lambda x: '% .3f' % x)
missing_df = pd.concat([missing_df, pct_df[pct_df > 0.0000]], axis=1, keys=['Total', 'Percent'])
missing_df.sort_values(axis=0, ascending=False, by=['Percent', 'Total'], inplace=True)
# missing_df['Percent'] = missing_df['Percent'].map(lambda x: '%.2f%%' % (x*100))
print(missing_df)

# drop feats that have over 70% missing values
# see if these fests have high correlation with y
for f in missing_df.index:
    try:
        print(f, corr_df.loc[f, 'SalePrice'])
    except KeyError:
        print(f, ': category feats')
# MasVnrArea:0.477493047096, LotFrontage:0.351799096571

wanna_drop_feats = list(missing_df[missing_df['Percent'] > 0.15].index)

# info that data description provided
fill_with_None = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                  'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                  'GarageCond', 'GarageQual', 'PoolQC', 'Fence', 'MiscFeature']
for f in fill_with_None:
    train[f].fillna('None', inplace=True)
    missing_df.drop(f, axis=0, inplace=True)

print(missing_df)
# Electrical: only have one missing value, drop this mising sample
train.drop(train.loc[train['Electrical'].isnull()].index, inplace=True)
missing_df.drop('Electrical', axis=0, inplace=True)
print('>> drop one sample due to missing value of Electrical')
# MasVnrArea
print(train.loc[train.loc[train['MasVnrArea'].isnull()].index, ['MasVnrType', 'MasVnrArea']])
# MasVnrArea is missing when MasVnrType is None, which means no MasVnrType, thus area is 0, so fill with 0
train['MasVnrArea'].fillna(0, inplace=True)
missing_df.drop('MasVnrArea', axis=0, inplace=True)
# LotFrontage: means that portion of a lot that abuts a public or private street.
# So it has something to do with Neighborhood and LotArea
train.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean', 'median', 'count'])
train['LotAreaCut'] = pd.qcut(train.LotArea, 10)
train.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean', 'median', 'count'])
train['LotFrontage'] = train.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
missing_df.drop('LotFrontage', axis=0, inplace=True)
print(train.isnull().sum().sum())
# why LotFrontage still has 7 missing values???
train['LotFrontage'] = train.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
print(train.isnull().sum().sum())  # 0
del train['LotAreaCut']

# cate_missing = [str(f) for f in missing_df.index if train.dtypes[f] == object and f not in fill_with_None]
# nume_missing = [str(f) for f in missing_df.index if train.dtypes[f] != object]

# # see if these feats should be kept
# for f in cate_missing:
#     plt.figure(figsize=(12, 8))
#     sns.boxplot(train[f], train.SalePrice)
#     train.groupby([f])[[f]].count()
#
# drop_cate_feats = ['basementfintype2', 'Fence']
# fill_cate_feats = list(set(cate_missing) ^ set(drop_cate_feats))
#
# # missing value of PoolQC
# plt.figure(figsize=(12, 8))
# sns.boxplot(train.PoolQC, train.SalePrice)
# train.groupby(['PoolQC'])[['PoolQC']].count()
# # it is clear that although PoolQC have so many missing values, it does effect price, so keep it
# train['PoolQC'].fillna('None', inplace=True)  # add a new value 'None' for 'PoolQC'
# wanna_drop_feats.remove('PoolQC')
#
# # missing value of MiscFeature, MiscVal(Miscellaneous feature not covered in other categories and its values)
# plt.figure(figsize=(12, 8))
# sns.boxplot(train.MiscFeature, train.SalePrice)
# train.groupby(['MiscFeature'])[['MiscFeature']].count()
# train['MiscFeature'].fillna('None', inplace=True)
# print(train['MiscFeature'].head())  # no missing value, the value of MiscVal where MiscFeature is None is 0
# wanna_drop_feats.remove('MiscFeature')
# # missing value of Alley
# plt.figure(figsize=(12, 8))
# sns.boxplot(train.Alley, train.SalePrice)
# train.groupby(['Alley'])[['Alley']].count()
# train['MiscFeature'].fillna('missing', inplace=True)
# wanna_drop_feats.remove('Alley')
#
# # Fence
# plt.figure(figsize=(12, 8))
# sns.boxplot(train.Fence, train.SalePrice)
# train.groupby(['Fence'])[['Fence']].count()
# train['Fence'].fillna('missing', inplace=True)
# wanna_drop_feats.remove('Fence')
#
# # FireplaceQu
# plt.figure(figsize=(12, 8))
# sns.boxplot(train.FireplaceQu, train.SalePrice)
# train.groupby(['FireplaceQu'])[['Fence']].count()
# train['FireplaceQu'].fillna('None', inplace=True)
# wanna_drop_feats.remove('FireplaceQu')
#
# # LotFrontage
# plt.figure(figsize=(12, 8))
# plt.scatter(train.LotFrontage, train.SalePrice)
#
# for f in wanna_drop_feats:
#     train.drop(f, inplace=True, axis=1)
#     print('> deleted from train due to missing value: %s ' % f)
# # ???: if keep and fill LotFrontage, MasVnrArea, will be any difference?


# feature engineer
print(train.dtypes[train.dtypes != object].index)

category_f = ['MSSubClass', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
for f in category_f:
    train[f] = train[f].astype('category')

# labelize
print(train.dtypes[train.dtypes == object].index)
# ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
# 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
# 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
# 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
# 'GarageType', 'GarageFinish', 'GarageQual','GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
# 'SaleType', 'SaleCondition']

# for f in train.dtypes[(train.dtypes == object) | (train.dtypes == 'category')].index:
#     # print(f, train[f].unique())
#     if len(train[f].unique()) > 5:  # may need re-encode
#         print(train.groupby(f)[['SalePrice']].agg(['mean', 'median', 'count'])
#               .sort_values(axis=0, ascending=False, by=('SalePrice', 'median')).plot(kind='bar'))

# re-encode some feats based on the above results and data——description
# cols1: need re-encode
cols1 = ['MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd']

# cols2: convert dtype, getdummy()

cols2 = [
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'GarageType', 'SaleType', 'SaleCondition', 'Condition1', 'Condition2', 'BsmtFinType2'
]

# cols3: do not need re-encode
cols3 = ['OverallQual', 'OverallCond']

# cols4:
cols4 = [
    'KitchenQual', 'BsmtQual', 'BsmtCond', 'BsmtFinType2', 'ExterQual', 'ExterCond', 'PoolQC', 'HeatingQC',
    'BsmtExposure', 'GarageQual', 'GarageCond', 'FireplaceQu'
]

# for cols4:


def encode_label(train):
    encoder = LabelEncoder()
    for col in cols4:
        train[col] = encoder.fit_transform(train[col])
        train[col] = train[col].astype('int')

    train["_MSZoning"] = encoder.fit_transform(train["MSZoning"])

    for col in ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']:
        train[col] = train[col].astype('int')
    for col in cols3:
        train[col] = train[col].astype('int')


# for cols1:


def map_values(train):
    train['_MSSubClass'] = train.MSSubClass.map({
        180: 1, 30: 1, 45: 1,
        190: 2, 50: 2, 90: 2, 85: 2, 40: 2, 160: 2,
        70: 3, 20: 3, 75: 3, 80: 3,
        120: 4, 60: 4
    })

    train['_Neighborhood'] = train.Neighborhood.map({
        'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1,
        'OldTown': 2, 'Edwards': 2, 'BrkSide':2,
        'Sawyer': 3, 'Blueste': 3, 'SWISU': 3, 'NAmes': 3, 'NPkVill': 3,
        'Mitchel': 4,
        'SawyerW': 5, 'Gilbert': 5, 'NWAmes': 5, 'Blmngtn': 5,
        'CollgCr': 6,
        'ClearCr': 7, 'Crawfor': 7,
        'Veenker': 8, 'Timber': 8,
        'Somerst': 9,
        'StoneBr': 10, 'NridgHt': 10,
        'NoRidge': 11
    })

    train['_Exterior1st'] = train.Exterior1st.map({
        'BrkComm': 1,
        'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
        'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3,
        'Stucco': 4, 'HdBoard': 4,
        'BrkFace': 5,
        'Plywood': 6,
        'VinylSd': 7,
        'CemntBd': 8,
        'Stone': 9, 'ImStucc': 9
    })

    train['_Exterior2nd'] = train.Exterior2nd.map({
        'CBlock': 1, 'AsbShng': 1,
        'Wd Sdng': 2, 'Wd Shng': 2, 'MetalSd': 2, 'AsphShn': 2, 'Stucco': 2,
        'Brk Cmn': 3,
        'HdBoard': 4, 'BrkFace': 4, 'Plywood': 4, 'Stone': 4,
        'ImStucc': 5,
        'VinylSd': 6,
        'CmentBd': 7,
        'Other': 8,
    })

    train['_Functional'] = train.Functional.map({
        'Maj2': 1,
        'Sev': 2,
        'Mod': 3,
        'Min1': 4, 'Min2': 4, 'Maj1': 4,
        'Typ': 5,
        'Sal': 6
    })


class LabelEnc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        encode_label(x)
        map_values(x)
        return x


# feature importance
class SkewDummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_numeric = x.select_dtypes(exclude=["object", "category"])
        skewness = x_numeric.apply(lambda t: skew(t))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        x[skewness_features] = np.log1p(x[skewness_features])
        x = pd.get_dummies(x)
        return x


# retain train for other use
train2 = train.copy()

# extract y
y = train2['SalePrice']
train2.drop('SalePrice', axis=1, inplace=True)
y_log = np.log(y)

# transform data
pipe = Pipeline([
    ('LabelEnc', LabelEnc()),
    ('SkewDummies', SkewDummies(skew=1)),
    ])
train_pipe = pipe.fit_transform(train2)
# train_pipe = pd.DataFrame(pipe.fit_transform(train2))
print('train_pipe: \n', train_pipe.head())

# split data
X_train, X_test, y_train, y_test = train_test_split(train_pipe, y_log, test_size=0.3, random_state=0)

rs = RobustScaler()
X_train = rs.fit_transform(X_train)
X_test = rs.transform(X_test)

# print(train_pipe.shape, X_train.shape, X_test.shape)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
FI_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=train_pipe.columns)
print("Feature Importance: \n", FI_lasso.sort_values("Feature Importance", ascending=False))
FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
plt.show()


# feature combination
class FeatCombine(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x["SUM_SF"] = x["TotalBsmtSF"] + x["BsmtFinSF1"] + x["WoodDeckSF"] + x["OpenPorchSF"] \
                      + x["ScreenPorch"] + x["3SsnPorch"] + x["EnclosedPorch"]
        x["SUM_Area"] = x["GrLivArea"] + x["PoolArea"] + x["MasVnrArea"] + x["LotArea"]

        x["SQ_SUM_SF"] = x["SUM_SF"] * x["SUM_SF"]
        x["SQ_SUM_Area"] = x["SUM_Area"] * x["SUM_Area"]
        
        x["SUM_Qu&Cond"] = x["OverallQual"] + x["OverallCond"] + x["KitchenQual"] + x["BsmtQual"] + x["BsmtCond"] \
                                            + x["BsmtFinType2"] + x["ExterQual"] + x["ExterCond"] + x["PoolQC"] \
                                            + x["HeatingQC"] + x["BsmtExposure"] + x["GarageQual"] + x["GarageCond"] \
                                            + x["FireplaceQu"]

        x['MIX_SF_Qu01'] = x["TotalBsmtSF"] * x["OverallQual"]
        x["MIX_Area_Qu01"] = x["GrLivArea"] * x["OverallQual"]

        x['MIX_SF_Qu02'] = x["SUM_SF"] * x["OverallQual"]
        x["MIX_Area_Qu02"] = x["SUM_Area"] * x["OverallQual"]

        # x["SUM_Room_Bath"] = x["FullBath"] + x["HalfBath"] + x["TotRmsAbvGrd"]

        x["MIX_Year_QU"] = x["YearBuilt"] * x["OverallQual"] / 100

        x["AVG_Year"] = x["YearBuilt"] + x["YearRemodAdd"] / 2

        x["MIX_Year_Area01"] = x["GrLivArea"] * x["YearBuilt"] / 100
        x["MIX_Year_Area02"] = x["GrLivArea"] * x["AVG_Year"] / 100
        x["MIX_Year_Area03"] = x["SUM_Area"] * x["YearBuilt"] / 100
        x["MIX_Year_Area04"] = x["SUM_Area"] * x["AVG_Year"] / 100

        x["MIX_Year_SF01"] = x["TotalBsmtSF"] * x["YearBuilt"] / 100
        x["MIX_Year_SF02"] = x["TotalBsmtSF"] * x["AVG_Year"] / 100
        x["MIX_Year_SF03"] = x["SUM_SF"] * x["YearBuilt"] / 100
        x["MIX_Year_SF04"] = x["SUM_SF"] * x["AVG_Year"] / 100

        x["MIX_Func_Area01"] = (x["_Functional"] + 1) * x["GrLivArea"]
        x["MIX_Func_SF01"] = (x["_Functional"] + 1) * x["TotalBsmtSF"]
        x["MIX_Func_Area02"] = (x["_Functional"] + 1) * x["SUM_Area"]
        x["MIX_Func_SF02"] = (x["_Functional"] + 1) * x["SUM_SF"]
        x["MIX_Func_Qu"] = (x["_Functional"] + 1) + x["OverallQual"]

        x["MIX_Neighbor_Area"] = x["_Neighborhood"] * x["SUM_SF"]
        x["MIX_Neighbor_Qu"] = x["_Neighborhood"] + x["OverallQual"]
        x["MIX_Neighbor_Year"] = x["_Neighborhood"] + x["YearBuilt"]

        x["MIX_MSZoning_SF"] = (x["_MSZoning"] + 1) * x["SUM_SF"]
        x["MIX_MSZoning_Qu"] = (x["_MSZoning"] + 1) + x["OverallQual"]
        x["MIX_MSZoning_Year"] = (x["_MSZoning"] + 1) + x["YearBuilt"]

        # x["AbnormalSymptoms"] = x["SaleCondition_Abnorml"] * x["SaleType"]

        return x


y = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)
y_log = np.log(y)
pipe = Pipeline([
    ('LabelEnc', LabelEnc()),
    ('FeatCombine', FeatCombine()),
    ('SkewDummies', SkewDummies(skew=1))
    ])

train_pipe = pipe.fit_transform(train)
print('train_pipe:\n', train_pipe.shape)

X_train, X_test, y_train, y_test = train_test_split(train_pipe, y_log, test_size=0.3, random_state=0)

rs = RobustScaler()
X_train = rs.fit_transform(X_train)
X_test = rs.transform(X_test)

pca = PCA(n_components=261)  # 259 is the feats numbers before feats combination
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("X_train and X_test Size: \n", X_train.shape, X_test.shape)


