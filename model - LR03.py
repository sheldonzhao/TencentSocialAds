# -*- coding: utf-8 -*-
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import f_pack
from sklearn.model_selection import train_test_split
import numpy as np

'''
LR03 = LR + num statistic features
[44.121425313]    connectionType_avg_cvr
[12.2195109973]        creativeID_avg_cvr
[20.4294180752]            gender_avg_cvr
[13.661466143]          haveBaby_avg_cvr
[13.4193468513]         residence_avg_cvr
[31.1306770954]         sitesetID_avg_cvr
[14.5878954593]          hometown_avg_cvr
xgboost 下采样可以借鉴
'''


def format_cross_features(dfTrain, dfTest, feat1, feat2):
    feat = feat1 + '-' + feat2
    dfTrain[feat] = dfTrain[feat1] + '.' + dfTrain[feat2]
    dfTrain[feat] = dfTrain[feat].astype(float)
    dfTrain[feat] = dfTrain[feat] * 10000
    dfTest[feat] = dfTest[feat1] + '.' + dfTest[feat2]
    dfTest[feat] = dfTest[feat].astype(float)
    dfTest[feat] = dfTest[feat] * 10000
    return dfTrain, dfTest


# load data
dfTrain, dfTest, y_label = f_pack.load_data()
# cross feature
feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
         "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'sitesetID', 'age', 'education', 'appID']
for feat in feats:
    dfTrain[feat], dfTest[feat] = dfTrain[feat].astype(str), dfTest[feat].astype(str)

dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'age', 'education')
dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appID')
dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appPlatform')

num_feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
             "appPlatform", 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'residence', 'appID',
             'positionType', 'sitesetID', 'hometown', 'appCategory']

for i, feat in enumerate(num_feats):
    dfCvr = dfTrain.groupby(feat).apply(lambda df: np.mean(df["label"])).reset_index()
    dfCvr.columns = [feat, feat + "_avg"]
    dfTrain = pd.merge(dfTrain, dfCvr, how="left", on=feat)
    dfTest = pd.merge(dfTest, dfCvr, how="left", on=feat)

dfTrain = dfTrain.fillna(0)
dfTrain = dfTrain.replace(np.inf, 0)
dfTest = dfTest.replace(np.inf, 0)
dfTest = dfTest.fillna(0)

enc = OneHotEncoder()
feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
         "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'age-education', 'positionID-appID',
         'positionID-appPlatform']
# 'age-education', 'positionID-appID', 'positionID-appPlatform','appID-sitesetID'
for i, feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

# merge num feats
for i, feat in enumerate(num_feats):
    X_train = sparse.hstack((X_train, dfTrain[feat + "_avg"].values.reshape(-1, 1)))
    X_test = sparse.hstack((X_test, dfTest[feat + "_avg"].values.reshape(-1, 1)))

X_train, test_set, y_train, y_test_set = train_test_split(X_train, y_label, test_size=0.2, random_state=0)
# model training
print("start modeling")
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:, 1]

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv(f_pack.file_submission, index=False)

# metric
predictions = lr.predict(test_set)
f_pack.print_metrics(y_test_set, predictions)
