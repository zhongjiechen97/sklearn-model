import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
# 1. 读取数据
data = pd.read_csv('./数据集.csv',index_col=0)

# print(data.info())
# 姓名          20 non-null object
# 搜索指数；日均值    20 non-null float64
# 咨询指数；日均值    20 non-null int64
# 年龄分布        20 non-null object
# 篇数          20 non-null int64
# 涉及公众号       20 non-null int64
# 阅读数         20 non-null int64
# 点赞数         20 non-null int64
# 10w+篇数      20 non-null int64
# 原创篇数        20 non-null int64
# 粉丝/万        20 non-null int64
# 微博数         20 non-null int64

# 2. 为标签编码，这里先简单的以第1列（搜索指数：日均值）作为标签
#    依据数值高低分为三类0、1、2，分别对应，不红、一般红、非常红

x = data.iloc[:,1].values.reshape(-1,1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
data.iloc[:,1] = est.fit_transform(x)

# 3. 为3列特征OneHot编码
y = data.iloc[:,3].values.reshape(-1,1)
enc = OneHotEncoder(categories="auto")
enc = enc.fit(y)
result = enc.transform(y).toarray()
# get_feature_names()获取OneHot编码的标签
# print(enc.get_feature_names())
# ['x0_30-39' 'x0_40-49']
newdata = pd.concat([data, pd.DataFrame(result)],axis=1)

newdata.drop(["姓名","年龄分布"],axis=1,inplace=True)
# 重新设置列标签
newdata.columns = [
    "搜索指数；日均值","咨询指数；日均值","年龄分布_30-39","年龄分布_40-49","篇数","涉及公众号","阅读数",
    "点赞数","10w+篇数","原创篇数","粉丝/万","微博数"
]

newdata.drop([0], axis=0, inplace=True)
newdata.fillna(0, inplace=True)

newdata = np.array(newdata)
data = list()
target = list()
for row in newdata:
    data.append(row)
for col in newdata:
    target.append(col[0])
target = np.array(target)
data = pd.DataFrame(data)
data.drop([0],axis=1,inplace=True)
data = np.array(data)

# print(data.shape)
# print(target.shape)
# (20, 11)
# (20,)

# 4. 寻找最好的n_components 结果为2
# pca_line = PCA().fit(data,target)
# pca_line.transform(data)
# plt.figure()
# plt.plot([1,2,3,4,5,6,7,8,9,10,11], np.cumsum(pca_line.explained_variance_ratio_))
# plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
# plt.xlabel("number of components after dimension reduction")
# plt.ylabel("cumulative explained variance")
# plt.legend()
# plt.show()
# 5. PCA降维
newdata = PCA(n_components=2).fit_transform(data,target)
pca = PCA(n_components=2)
pca = pca.fit(data,target)
newdata = pca.transform(data)
data = newdata
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())
# print(data.shape)
# (20, 2)

# 6. 寻找最佳n_estimators
results = []
for i in range(40,100):
    rfc = RandomForestRegressor(n_estimators=i,n_jobs=-1)
    rfc_s = cross_val_score(rfc,data,target,cv=10).mean()
    results.append(rfc_s)
print("Score最大值{:.4}:，对应的n_estimators:{:}".format(max(results),40+(results.index(max(results)))))
plt.figure(figsize=[100,20])
plt.plot(range(40,100),results)
plt.show()
# Score最大值0.6972:，对应的n_estimators:55

