# -*- coding: utf-8 -*-
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import f_pack
from sklearn.model_selection import train_test_split


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
         "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'avg_cvr', 'age', 'education', 'appID',
         'sitesetID']
for feat in feats:
    dfTrain[feat], dfTest[feat] = dfTrain[feat].astype(str), dfTest[feat].astype(str)

dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'age', 'education')
dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appID')
dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appPlatform')

enc = OneHotEncoder()
feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
         "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'age', 'education',
         'avg_cvr', 'positionID-appID', 'positionID-appPlatform', 'age-education']

for i, feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

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
