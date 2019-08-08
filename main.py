import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def _load_data():
    train_data = pd.read_csv('train_set.csv')
    test_data = pd.read_csv('test_set.csv')

    def preprocessing(df):

        df.loc[df['pdays'] == -1, 'pdays'] = 999  # 距离上次最后一次活动没有联系过的用户
        df.loc[df['poutcome'] == 'other', 'poutcome'] = 'unknown'  # 不确定与其他等价
        return df

    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)

    cols = [c for c in train_data.columns if c not in ['ID', 'y']]
    cols = [c for c in cols if (not c in IGNORE_COLS)]  # 得到所有的特征列表名称
    x_train = train_data[cols].values
    y_train = train_data['y'].values
    x_test = test_data[cols].values
    ids_test = test_data['ID'].values
    cat_feature_indices = [i for i, c in enumerate(cols) if c in CATEGORICAL_COLS]

    return train_data, test_data, x_train, y_train, x_test, ids_test, cat_feature_indices


def FeatureDictionary(dfTrain, dfTest, numeric_cols, ignore_cols):
    """
    parameters：
               dfTrain,dfTest:训练、测试数据集
               numeric_cols,ignore_cols：数值型、忽略的列名
    return：
           feat_dic：数值型变量名以及类别型变量类别对应的字典
           feat_dim：总个数
    """
    df = pd.concat([dfTrain, dfTest])
    feat_dic = {}
    tc = 0
    for col in df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            feat_dic[col] = tc
            tc += 1
        else:
            us = df[col].unique()
            feat_dic[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
    feat_dim = tc

    return feat_dic, feat_dim


def data_parser(df, feat_dic, categorical_cols, ignore_cols, has_label=False):
    """
    parameters：
               df：数据；
               feat_dic:数值型变量名以及类别型变量类别对应的字典
    return：
               Xi,y：训练数据
               Xi，ids：测试数据
    """
    dfi = df.copy()
    if has_label:
        y = dfi['y'].values.tolist()
        dfi.drop(['ID', 'y'], axis=1, inplace=True)
    else:
        ids = dfi['ID'].values.tolist()
        dfi.drop(['ID'], axis=1, inplace=True)

    for col in dfi.columns:
        if col in ignore_cols:
            dfi.drop(col, axis=1, inplace=True)
            continue
        if col in categorical_cols:
            dfi[col] = dfi[col].map(feat_dic[col])
    Xi = dfi.values.tolist()
    if has_label:
        return np.array(Xi), np.array(y)
    else:
        return np.array(Xi), np.array(ids)

#Logistic Regression
from sklearn.linear_model import LogisticRegression


def Logistic_Regression(Xi_train_, Xi_test_, y_train_, y_test_):

    sta = time.time()
    classifier = LogisticRegression(solver='lbfgs',max_iter=100,multi_class='multinomial')
    classifier.fit(Xi_train_,y_train_)
    # y_pred = classifier.predict_proba(Xi_test)[:,1]
    # result = pd.DataFrame(columns=['ID','pred'])
    # result['ID'] = ids_test.tolist()
    # result['pred'] = y_pred.tolist()
    # result.to_csv('result_LR.csv')
    y_pred = classifier.predict_proba(Xi_test_)[:, 1]
    score = roc_auc_score(y_test_, y_pred)
    end = time.time()
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score))
    print('time:{0:.1f}s'.format(end - sta))
    return score

#SGDClassifier
from sklearn.linear_model import SGDClassifier


def SGD(Xi_train_, Xi_test_, y_train_, y_test_):
    sta = time.time()
    classifier_SGD = SGDClassifier(loss='log', penalty='l2', alpha=0.0003, max_iter=200, learning_rate='optimal')
    classifier_SGD.fit(Xi_train_, y_train_)
    y_pred_SGD = classifier_SGD.predict_proba(Xi_test_)[:, 1]
    score_SGD = roc_auc_score(y_test_, y_pred_SGD)
    end = time.time()
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score_SGD))
    print('time:{0:.1f}s'.format(end - sta))
    return  score_SGD

#SVM
from sklearn.svm import SVC


def SVMClassifier(Xi_train_, Xi_test_, y_train_, y_test_):
    sta = time.time()
    classifier_SVM = SVC(C=0.5, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                         max_iter=-1, probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False)
    classifier_SVM.fit(Xi_train_, y_train_)
    y_pred_SVM = classifier_SVM.predict(Xi_test_)
    score_SVM = roc_auc_score(y_test_, y_pred_SVM)
    end = time.time()
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score_SVM))
    print('time:{0:.1f}s'.format(end - sta))
    return score_SVM

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def Decision_Trees(Xi_train_, Xi_test_, y_train_, y_test_):
    sta = time.time()
    # Train decision tree classifier
    params = {'max_depth': [3, 10, None]}
    decision_tree_model = DecisionTreeClassifier(criterion='gini',min_samples_split=30)
    grid_search = GridSearchCV(decision_tree_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search.fit(Xi_train_, y_train_)

    # Use model with best parameter as final model
    decision_tree_final = grid_search.best_estimator_

    # Evaluate and run model on training data
    y_pred_DT = decision_tree_final.predict_proba(Xi_test_)[:, 1]
    score_DT = roc_auc_score(y_test_, y_pred_DT)
    end = time.time()
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score_DT))
    print('time:{0:.1f}s'.format(end - sta))
    return  score_DT

#random forest
from sklearn.ensemble import RandomForestClassifier


def Random_Forest(Xi_train_, Xi_test_, y_train_, y_test_):
    sta = time.time()
    params = {'max_depth': [3, 10, None]}
    random_forest_model = RandomForestClassifier(n_estimators=70, criterion='gini', min_samples_split=15,
                                                 n_jobs=-1)
    grid_search = GridSearchCV(random_forest_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search.fit(Xi_train_, y_train_)

    # Use best paramter for final model
    random_forest_final = grid_search.best_estimator_

    # Evaluate model
    # y_pred_RF = random_forest_final.predict_proba(Xi_test)[:, 1]
    # result = pd.DataFrame(columns=['ID','pred'])
    # result['ID'] = ids_test.tolist()
    # result['pred'] = y_pred_RF.tolist()
    # result.to_csv('result.csv')

    # Evaluate model
    y_pred_RF = random_forest_final.predict_proba(Xi_test_)[:, 1]
    score_RF = roc_auc_score(y_test_, y_pred_RF)
    end = time.time()
    print('✔  ROC AUC score on test set: {0:.3f}'.format(score_RF))
    print('time:{0:.1f}s'.format(end - sta))
    return  score_RF


IGNORE_COLS = ['day','month']
CATEGORICAL_COLS = ['job','marital','education','default','housing','loan','contact','poutcome']
NUMERIC_COLS = ['age','balance','duration','campaign','pdays','previous']

train_data,test_data,x_train,y_train,x_test,ids_test,cat_feature_indices =  _load_data()
feat_dic,feat_dim = FeatureDictionary(train_data,test_data,NUMERIC_COLS,IGNORE_COLS)
Xi_train,y_train = data_parser(train_data,feat_dic,CATEGORICAL_COLS,IGNORE_COLS,has_label=True)
Xi_test,ids_test = data_parser(test_data,feat_dic,CATEGORICAL_COLS,IGNORE_COLS,has_label=False)

#Feature Scaling
sc = StandardScaler()
Xi_train = sc.fit_transform(Xi_train)
Xi_test = sc.fit_transform(Xi_test)

#splitting the data Training and Test
print(Xi_train.shape)
Xi_train_, Xi_test_, y_train_, y_test_ = train_test_split(Xi_train,y_train,test_size=.30, random_state=3)

results = []
result1 = Logistic_Regression(Xi_train_, Xi_test_, y_train_, y_test_)
results += [result1]

result2 = SGD(Xi_train_, Xi_test_, y_train_, y_test_)
results += [result2]
result3 = SVMClassifier(Xi_train_, Xi_test_, y_train_, y_test_)
results += [result3]
result4 = Decision_Trees(Xi_train_, Xi_test_, y_train_, y_test_)
results += [result4]
result5 = Random_Forest(Xi_train_, Xi_test_, y_train_, y_test_)
results += [result5]

result_data = pd.Series(results, index=['LR','SGD','SVM','DT','RF'])
result_data.plot.bar()
plt.show()

