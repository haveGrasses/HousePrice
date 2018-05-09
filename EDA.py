
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
train = pd.read_csv('./input/train.csv')


# data explore
print(train.head())
# Id column is useless
train.drop('Id', axis=1, inplace=True)
print(train.columns, train.dtypes)
print(train.describe())


# correlation analysis
corr_df = train.select_dtypes(exclude=['object']).corr()
print(corr_df.head())

# correlation with y
corr_with_y = corr_df['SalePrice'].to_dict()
del corr_with_y['SalePrice']
print(' the correlation of numerical features and SalesPrice:\n')
for ele in sorted(corr_with_y.items(), key=lambda x: -abs(x[1])):
    print('{0}: {1}'.format(*ele))

# mutual correlation between features
high_corr_list = []
feats = set()
for i, corr in corr_df[(corr_df >= 0.7) & (corr_df != 1.0)].to_dict().items():
    for k, v in corr.items():
        if pd.isnull(v):  # 此处有坑：如果用 if v == np.nan来判断，如果v是nan也会返回False，因为nan是float的子类
            # 正确的判断方法：np.isnan(), pd.isnull(), import math math.isnan()
            continue
        elif i != 'SalePrice' and k != 'SalePrice':
            # print(i, k, v)
            feats.add(set((i, k)))
            if set((i, k)) not in feats:
                high_corr_list.append((i, k, v))
high_corr_df = pd.DataFrame(high_corr_list, columns=['feat1', 'feat2', 'feat1 vs feat2'])
del high_corr_list
gc.collect()
feat1_vs_y, feat2_vs_y = [], []
for f in high_corr_df['feat1']:
    feat1_vs_y.append(corr_df.loc[f, 'SalePrice'])
for f in high_corr_df['feat2']:
    feat2_vs_y.append(corr_df.loc[f, 'SalePrice'])
high_corr_df['feat1 vs y'] = feat1_vs_y
high_corr_df['feat2 vs y'] = feat2_vs_y

gc.collect()
print(high_corr_df)
# heatmap visual
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

