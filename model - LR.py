# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import f_pack

# load data
dfTrain = pd.read_csv(f_pack.file_train)
# dfTrain = dfTrain[dfTrain['clickTime'] <= 290000]
dfTest = pd.read_csv(f_pack.file_test)
dfAd = pd.read_csv(f_pack.file_ad)
dfUser = pd.read_csv(f_pack.file_user)
dfUser['age'] = dfUser['age'].map(lambda x: (x // 5) + 1 if x > 0 else 0)
dfUser['haveBaby'] = dfUser['haveBaby'].map(lambda x: 3 if x >= 3 else x)
dfAppCate = pd.read_csv(f_pack.file_app_categories)
dfPos = pd.read_csv(f_pack.file_position)
# process data
print(dfTrain.shape)
print(dfTest.shape)
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTrain = pd.merge(dfTrain, dfUser, on='userID')
dfTest = pd.merge(dfTest, dfUser, on='userID')
dfTrain = pd.merge(dfTrain, dfAppCate, on='appID')
dfTest = pd.merge(dfTest, dfAppCate, on='appID')
dfTrain = pd.merge(dfTrain, dfPos, on='positionID')
dfTest = pd.merge(dfTest, dfPos, on='positionID')
y_train = dfTrain["label"].values
print(dfTrain.shape)
print(dfTest.shape)

# feature engineering/encoding
enc = OneHotEncoder()
feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
         "appID", "appPlatform", 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'residence', 'appCategory']
for i, feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

# model training
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:, 1]

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv(f_pack.file_submission, index=False)
