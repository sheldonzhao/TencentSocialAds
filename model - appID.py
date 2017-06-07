"""
baseline 1: history pCVR of creativeID/adID/camgaignID/advertiserID/appID/appPlatform
"""

# res: 0.1 appID is a great feature

import zipfile
import numpy as np
import pandas as pd
import f_pack

# load data
dfTrain = pd.read_csv(f_pack.file_train)
dfTest = pd.read_csv(f_pack.file_test)
dfAd = pd.read_csv(f_pack.file_ad)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
y_train = dfTrain["label"].values

# model building
key = "appID"
dfCvr = dfTrain.groupby(key).apply(lambda df: np.mean(df["label"])).reset_index()
dfCvr.columns = [key, "avg_cvr"]
dfCvr.to_csv(f_pack.file_appID_score)
dfTest = pd.merge(dfTest, dfCvr, how="left", on=key)
dfTest["avg_cvr"].fillna(np.mean(dfTrain["label"]), inplace=True)
proba_test = dfTest["avg_cvr"].values

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv(f_pack.file_submission, index=False)
