
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

pd.set_option('display.width', 500, 'display.max_rows', 500, 'precision', 2)

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
plt.show()
# pair plot
k = 5
cols = corr_df.nlargest(k, 'SalePrice')['SalePrice'].index
sns.set()
sns.pairplot(train[cols])
plt.show()


# outliers

