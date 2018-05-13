
import os
os.chdir('F:\python\machine learning\HousePrice')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc


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
cate_missing = [str(f) for f in missing_df.index if train.dtypes[f] == object]
nume_missing = [str(f) for f in missing_df.index if train.dtypes[f] != object]

# see if these feats should be kept
for f in cate_missing:
    plt.figure(figsize=(12, 8))
    sns.boxplot(train[f], train.SalePrice)
    train.groupby([f])[[f]].count()

drop_feats = ['basementfintype2', 'Fence']

# missing value of PoolQC
plt.figure(figsize=(12, 8))
sns.boxplot(train.PoolQC, train.SalePrice)
train.groupby(['PoolQC'])[['PoolQC']].count()
# it is clear that although PoolQC have so many missing values, it does effect price, so keep it
train['PoolQC'].fillna('None', inplace=True)  # add a new value 'None' for 'PoolQC'
wanna_drop_feats.remove('PoolQC')

# missing value of MiscFeature, MiscVal(Miscellaneous feature not covered in other categories and its values)
plt.figure(figsize=(12, 8))
sns.boxplot(train.MiscFeature, train.SalePrice)
train.groupby(['MiscFeature'])[['MiscFeature']].count()
train['MiscFeature'].fillna('None', inplace=True)
print(train['MiscFeature'].head())  # no missing value, the value of MiscVal where MiscFeature is None is 0
wanna_drop_feats.remove('MiscFeature')
# missing value of Alley
plt.figure(figsize=(12, 8))
sns.boxplot(train.Alley, train.SalePrice)
train.groupby(['Alley'])[['Alley']].count()
train['MiscFeature'].fillna('missing', inplace=True)
wanna_drop_feats.remove('Alley')

# Fence
plt.figure(figsize=(12, 8))
sns.boxplot(train.Fence, train.SalePrice)
train.groupby(['Fence'])[['Fence']].count()
train['Fence'].fillna('missing', inplace=True)
wanna_drop_feats.remove('Fence')

# FireplaceQu
plt.figure(figsize=(12, 8))
sns.boxplot(train.FireplaceQu, train.SalePrice)
train.groupby(['FireplaceQu'])[['Fence']].count()
train['FireplaceQu'].fillna('None', inplace=True)
wanna_drop_feats.remove('FireplaceQu')

# LotFrontage
plt.figure(figsize=(12, 8))
plt.scatter(train.LotFrontage, train.SalePrice)


for f in wanna_drop_feats:
    train.drop(f, inplace=True, axis=1)
    print('> deleted from train due to missing value: %s ' % f)
# ???: if keep and fill LotFrontage, MasVnrArea, will be any difference?

# Electrical only have one missing value, drop this mising sample
train.drop(train.loc[train['Electrical'].isnull()].index, inplace=True)
print('>> drop one sample due to missing value of Electrical')

