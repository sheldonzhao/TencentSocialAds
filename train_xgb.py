from feature_set import make_train_set
from feature_set import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime
import pandas as pd
from sklearn.utils import resample
import f_pack
import matplotlib.pyplot as plt
import operator

def statistic(group):
    group = group.sort_values('label', ascending=False).head(1)
    return group

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def underSampling(training_data, label):
    n = label.size // 8
    small_data, small_label = resample(training_data, label, n_samples=n)
    data = pd.concat([training_data, label], axis=1)
    positive = data[data['label'] == 1]
    small_data = pd.concat([small_data, positive.ix[:, :positive.columns.size - 1]])
    small_label = pd.concat([small_label, positive.ix[:, positive.columns.size - 1]])
    return small_data, small_label


def xgboost_make_submission():
    training_data, label = make_train_set()
    instanceID, test_trainning_data = make_test_set()
    instanceID = instanceID.reset_index()
    test_trainning_data = xgb.DMatrix(test_trainning_data.values)
    print('start fit')
    for i in range(1):
        small_data, small_label = underSampling(training_data, label)
        ceate_feature_map(small_data.columns)
        X_train, X_test, y_train, y_test = train_test_split(small_data.values, small_label.values, test_size=0.2,
                                                            random_state=0)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
                 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
                 'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
        num_round = 300
        watchList = [(dtest, 'eval'), (dtrain, 'train')]
        plst = list(param.items()) + [('eval_metric', 'logloss')]
        bst = xgb.train(plst, dtrain, num_round, watchList)
        y = bst.predict(test_trainning_data)
        instanceID = pd.concat([instanceID, pd.Series(y)], axis=1)
        # feature importance
        feature_score=bst.get_fscore(fmap='xgb.fmap')
        print(feature_score)
        feature_score = sorted(feature_score.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(feature_score, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
        plt.show()
    instanceID.to_csv(f_pack.file_output_test, index=False)

    # output
    data = f_pack.read_file(f_pack.file_output_test)
    data = data.ix[:, 1:]
    print(data.head())
    data['Prob'] = data.ix[:, 1:].sum(axis=1) / 10
    data = data[['instanceID', 'Prob']]
    data.to_csv(f_pack.file_submission, index=False)


if __name__ == '__main__':
    print(datetime.now())
    xgboost_make_submission()
    print(datetime.now())
