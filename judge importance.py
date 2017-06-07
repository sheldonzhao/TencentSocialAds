# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
import f_pack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def format_cross_features(dfTrain, feat1, feat2):
    feat = feat1 + '-' + feat2
    dfTrain[feat] = dfTrain[feat1] + '.' + dfTrain[feat2]
    dfTrain[feat] = dfTrain[feat].astype(float)
    dfTrain[feat] = dfTrain[feat] * 10000
    return dfTrain


def cross_features(feat1, feat2):
    # load data
    dfTrain, y_label = f_pack.get_training_set()
    # cross feature
    for feat in dfTrain.columns:
        dfTrain[feat] = dfTrain[feat].astype(str)
    dfTrain = format_cross_features(dfTrain, feat1, feat2)

    enc = OneHotEncoder()
    feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
             "appPlatform", 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'residence', 'avg_cvr']
    feats += [feat1 + '-' + feat2]
    print(feats)
    for i, feat in enumerate(feats):
        x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
        if i == 0:
            X_train = x_train
        else:
            X_train = sparse.hstack((X_train, x_train))

    X_train, test_set, y_train, y_test_set = train_test_split(X_train, y_label, test_size=0.2, random_state=0)
    # model training
    print("start modeling")
    lr = LogisticRegression(penalty='l1')
    lr.fit(X_train, y_train)

    # metric
    predictions = lr.predict(test_set)
    f_pack.print_metrics(y_test_set, predictions)
    print('---------------------------------------')


if __name__ == '__main__':
    print(datetime.now())
    cross_features('appPlatform', 'sitesetID')
    print(datetime.now())
