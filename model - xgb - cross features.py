import pandas as pd
import f_pack
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import xgboost as xgb
import operator


def underSampling(training_data, label):
    n = label.size // 10
    small_data, small_label = resample(training_data, label, n_samples=n)
    positive = training_data[training_data['label'] == 1]
    small_data = pd.concat([small_data, positive])
    small_label = pd.concat([pd.DataFrame(small_label), positive['label']])
    return small_data, small_label


def format_cross_features(dfTrain, dfTest, feat1, feat2):
    feat = feat1 + '-' + feat2
    dfTrain[feat] = dfTrain[feat1] + '.' + dfTrain[feat2]
    dfTrain[feat] = dfTrain[feat].astype(float)
    dfTrain[feat] = dfTrain[feat] * 10000
    dfTest[feat] = dfTest[feat1] + '.' + dfTest[feat2]
    dfTest[feat] = dfTest[feat].astype(float)
    dfTest[feat] = dfTest[feat] * 10000
    return dfTrain, dfTest


def model(n_round):
    # load data
    dfTrain, dfTest, y_label = f_pack.load_data()
    # cross feature
    feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "sitesetID", "advertiserID",
             "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'age', 'education', 'appID']
    for feat in feats:
        dfTrain[feat], dfTest[feat] = dfTrain[feat].astype(str), dfTest[feat].astype(str)

    dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'age', 'education')
    dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appID')
    dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'appPlatform')
    dfTrain, dfTest = format_cross_features(dfTrain, dfTest, 'positionID', 'connectionType')

    dfTrain = dfTrain.fillna(0)
    dfTrain = dfTrain.replace(np.inf, 0)
    dfTest = dfTest.replace(np.inf, 0)
    dfTest = dfTest.fillna(0)

    enc = OneHotEncoder()
    feats = ['positionID', 'connectionType', 'telecomsOperator', "creativeID", "adID", "camgaignID", "advertiserID",
             "appPlatform", 'gender', 'marriageStatus', 'haveBaby', 'residence', 'age', 'education', 'positionID-appID',
             'positionID-appPlatform', 'positionID-connectionType']

    for i, feat in enumerate(feats):
        x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
        x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

    feats = ['download_num', 'avg_cvr', 'user_install_num', 'app_install_num', 'h_potential', 'h_potential02']
    for feat in feats:
        X_train = sparse.hstack((X_train, dfTrain[feat].values.reshape(-1, 1)))
        X_test = sparse.hstack((X_test, dfTest[feat].values.reshape(-1, 1)))

    # model training
    print("start modeling")
    X_train, valid_set, y_train, y_valid = train_test_split(X_train, y_label, test_size=0.05, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(valid_set, label=y_valid)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 10,
             'min_child_weight': 5, 'gamma': 0, 'silent': 1, 'objective': 'binary:logistic',
             'early_stopping_rounds': 50}
    # xgb.cv(param, dtrain, n_round, nfold=5, metrics={'auc'}, seed=0,
    #      callbacks=[xgb.callback.print_evaluation(show_stdv=True)], early_stopping_rounds=20)
    watchList = [(dtest, 'eval'), (dtrain, 'train')]
    plst = list(param.items()) + [('eval_metric', 'logloss')]
    bst = xgb.train(plst, dtrain, n_round, watchList)
    y = bst.predict(xgb.DMatrix(X_test))
    res = pd.concat([dfTest['instanceID'], pd.Series(y)], axis=1)
    res = res.sort_values('instanceID')
    res['instanceID'] = res['instanceID'].astype(int)
    res.columns = ['instanceID', 'proba']
    print(res.shape)
    res.to_csv(f_pack.file_submission, index=False)


model(230)

'''
# feature importance
feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=operator.itemgetter(1))
print(feature_score)
df = pd.DataFrame(feature_score, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.show()

proba_test = lr.predict_proba(X_test)[:, 1]
# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv(f_pack.file_submission, index=False)
'''
