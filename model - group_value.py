import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import f_pack

# load data
dfTrain = pd.read_csv(f_pack.file_train)
dfTest = pd.read_csv(f_pack.file_test)
dfAd = pd.read_csv(f_pack.file_ad)
dfUser = pd.read_csv(f_pack.file_user)
dfUser['age'] = dfUser['age'].map(lambda x: (x // 5) + 1 if x > 0 else 0)
dfUser['haveBaby'] = dfUser['haveBaby'].map(lambda x: 3 if x >= 3 else x)
dfAppCate = pd.read_csv(f_pack.file_app_categories)
dfPos = pd.read_csv(f_pack.file_position)
dfAppScore = pd.read_csv(f_pack.file_appID_score)
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
feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
         "appPlatform", 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'residence', 'appID',
         'positionType', 'sitesetID', 'hometown', 'appCategory']

for i, feat in enumerate(feats):
    dfCvr = dfTrain.groupby(feat).apply(lambda df: np.mean(df["label"])).reset_index()
    dfCvr.columns = [feat, feat + "_avg_cvr"]
    dfTrain = pd.merge(dfTrain, dfCvr, how="left", on=feat)
    dfTest = pd.merge(dfTest, dfCvr, how="left", on=feat)
dfTest.fillna(0, inplace=True)

filt = '.*_avg_cvr'
X_train = dfTrain.filter(regex=filt)
X_test = dfTest.filter(regex=filt)

# model building
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:, 1]
param=pd.DataFrame({"columns":list(X_train.columns), "coef":list(lr.coef_.T)})
print(param)

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv(f_pack.file_submission, index=False)
