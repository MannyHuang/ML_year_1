#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:45:12 2018

@author: Manny
"""

# Importing the libraries
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
#from imblearn.over_sampling import RandomOverSampler
from collections import Counter
#from imblearn.combine import SMOTEENN
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb 
import operator 
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance

# get the ratio of high risk customers
def get_highrisk_ratio(tag_list):
    Y = tag_list
    y1_count = Y.count(1)
    ratio = y1_count/len(Y)
    print('highrisk customers account for :', ratio)
    return ratio
    
# print shapes of all raw data
def print_data_shapes(train_xy, train_x, test_all, train_withlabel):
    print(train_xy.shape)
    print(train_xy.info())
    
    print(train_x.shape)
    print(train_x.info())
    
    print(test_all.shape)
    print(test_all.info())

    print(train_withlabel.describe())
    

# compute baseline for labelled data
def find_baseline_labelled(train_withlabel, out_dir):
    sns.countplot(x='y', data=train_withlabel)
    test_all['y'] = 0
    test_all[['cust_id', 'y']].to_csv(os.path.join(out_dir, 'priditction.csv'), index=False)    

def plot_feature(test, train_withlabel):    
    # Concatenating training and test sets
    data = pd.concat([train_withlabel, test], axis=1)
    
    # Comparing feature selection
    E = np.random.uniform(0, 0.1, size=(len(data.values), 20))
    X_ = np.hstack((features.values, E))
    
    
    plt.figure(1)
    plt.clf()
    
    X_indices = np.arange(X_.shape[-1])
    
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X_, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
            edgecolor='black')
    
    clf = svm.SVC(kernel='linear')
    clf.fit(X_, y)
    
    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()
    
    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
            color='navy', edgecolor='black')
    
    clf_selected = svm.SVC(kernel='linear')
    clf_selected.fit(selector.transform(X_), y)
    
    svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.max()
    
    plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection', color='c',
            edgecolor='black')
    
    
    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

# get the indices of selected features
def get_feature_indices(data_set, feature_set):
    data_set = data_set.values
    fea_index = []
    for A_col in np.arange(data_set.shape[1]):
        for B_col in np.arange(feature_set.shape[1]):
            if (data_set[:, A_col] == feature_set[:, B_col]).all():
                fea_index.append(A_col)
    return fea_index

def cv_parameter_tuning(clf, features, y):
    param_test = {
        'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
        'n_estimators': [50, 200, 50],
        'learning_rate': [0.01, 0.03, 0.05, 0.07],
        'num_leaves': [20, 30, 40],
        'max_depth': [4, 6, 8],
        'reg_alpha': [0.0, 0.02, 0.04],
        'reg_lambda': [0.0, 0.03, 0.06],
        'min_split_gain': [0, 0.02],
        'min_child_weight': [0.001, 0.5, 10, 40]
    }
    gsearch = GridSearchCV(clf, param_grid=param_test, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch.fit(features, y)
    print(gsearch.best_params_)


def train_result(features, y, times):
    Accuracy = 0.0
    AUC = 0.0
    for time in range(times):
        xy_x_train, xy_x_test, xy_y_train, xy_y_test = train_test_split(features, y, test_size=0.4)
        """n_jobs=-1,
          n_estimators=100,
          learning_rate=0.1,
          num_leaves=20,
          max_depth=4,
          reg_alpha=0.00,
          reg_lambda=0.00,
          # boosting_type='rf'"""
        LGBM = LGBMClassifier(learning_rate=0.01, min_split_gain=0.02, reg_alpha=0.04,
                              min_child_weight=0, reg_lambda=0.07, n_estimators=200, max_depth=4, num_leaves=20
                              )
        LGBM.fit(x_train, y_train)
        y_pred = LGBM.predict(x_test)
        y_predict = LGBM.predict_proba(x_test)[:, 1]
        Accuracy += metrics.accuracy_score(y_test, y_pred)
        AUC += metrics.roc_auc_score(y_test, y_predict)
    Accuracy /= times
    AUC /= times
    print("Accuracy : %.4g" % Accuracy)
    print("AUC Score (Train): %f" % AUC)


def my_plot_importance(booster, figsize, **kwargs): 
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)
    






'''
Loading Data
'''

# prepare directories
base_path = r"C:\Users\palad\Desktop\ML_year_1"
os.chdir(base_path)
data_dir = "data"
out_dir = "output"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# import dataset
train_x = pd.read_csv(os.path.join(data_dir, 'train_x.csv'))
train_xy = pd.read_csv(os.path.join(data_dir, 'train_xy.csv'))
test_all = pd.read_csv(os.path.join(data_dir, 'test_all.csv'))

# remove irrelevent columns
train_withlabel = train_xy.drop(['y', 'cust_id', 'cust_group'], axis=1)
train_nolabel = train_x.drop(['cust_id', 'cust_group'], axis=1)
test = test_all.drop(['cust_id', 'cust_group'], axis=1)
Y = list(train_xy['y'] )    



# replace missing values with nan
#train_withlabel.replace({-99:np.nan}, inplace=True)
#train_xy.replace({-99:np.nan}, inplace=True)
#test.replace({-99:np.nan}, inplace=True)





'''
Analyze missing data
'''

# sort based on the ratio of missing data
#sort_total_missing_data = train_withlabel.isnull().sum().sort_values(ascending=False)
#sort_percent = (train_withlabel.isnull().sum()/train_withlabel.isnull().count()).sort_values(ascending = False)
#sort_missing_data = pd.concat([sort_total_missing_data, sort_percent], axis = 1, keys = ['Total_missing_data', 'Percent'])

# Ploting the ratio of missing data
#total_missing_data = train_withlabel.isnull().sum()
#percent = train_withlabel.isnull().sum()/train_withlabel.isnull().count()
#missing_data = pd.concat([total_missing_data, percent], axis = 1, keys = ['Total_missing_data', 'Percent'])
#missing_data = missing_data['Percent'].values

# sort missing value by raw
#missing_value_feature = train_withlabel.shape[1] - train_withlabel.count(axis=1)
#train_xy['missing_values'] = missing_value_feature
#train_withlabel['missing_values'] = missing_value_feature
#stat = train_xy[['y', 'missing_values']]

#stat.plot.scatter(x='y', y='missing_values')

#np.savetxt(os.path.join(out_dir, 'stat.csv'), stat, delimiter=",")



'''
feature engineering
'''

# Dropping features with more than 99% missing values according to sort_missing_data
numerical_features = train_withlabel.iloc[:,0:95]
categorical_features = train_withlabel.iloc[:,95:]
train_nolabel = train_nolabel.iloc[:,:]

numerical_features.drop(['x_94','x_92'],inplace = True,axis = 1)
categorical_features.drop(['x_129','x_132','x_134','x_116','x_110','x_112','x_118','x_113','x_126','x_131',
               'x_133','x_135','x_137','x_114','x_138','x_102','x_123','x_125','x_107','x_130',
               'x_119','x_128','x_115','x_109','x_117','x_127','x_103','x_111','x_108','x_136',
               'x_124','x_104','x_106','x_120','x_122','x_121','x_105'],inplace = True,axis = 1)   

test.drop(['x_94','x_92','x_129','x_132','x_134','x_116','x_110','x_112','x_118','x_113','x_126','x_131',
           'x_133','x_135','x_137','x_114','x_138','x_102','x_123','x_125','x_107','x_130','x_119','x_128',
           'x_115','x_109','x_117','x_127','x_103','x_111','x_108','x_136', 'x_124','x_104','x_106',
           'x_120','x_122','x_121','x_105'],inplace = True,axis = 1)

train_nolabel.drop(['x_94','x_92','x_129','x_132','x_134','x_116','x_110','x_112','x_118','x_113','x_126','x_131',
           'x_133','x_135','x_137','x_114','x_138','x_102','x_123','x_125','x_107','x_130','x_119','x_128',
           'x_115','x_109','x_117','x_127','x_103','x_111','x_108','x_136', 'x_124','x_104','x_106',
           'x_120','x_122','x_121','x_105'],inplace = True,axis = 1)    
    
    
'''
preprocessing
'''    

# impute missing numerical variables with mean
#for i in numerical_features:
#    numerical_features[i] = numerical_features[i].fillna(numerical_features[i].mean())   

 # Imputing missing caterogical variables
#categorical_features = categorical_features.fillna(0)

# Imputing missing caterogical variables with KNN complaint about too much missing values
#knnOutput = KNN(k=5).complete(categorical_features)

categorical_data = pd.concat([categorical_features, train_xy['y']], axis=1)
categorical_data.to_csv(os.path.join(out_dir, 'categorical_data.csv'), index=False)


# Concatenating numerical_features and categorical_features
features = pd.concat([numerical_features, categorical_features], axis=1)

# Split original training data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    features.values, Y, test_size=0.33, random_state=42, stratify=Y) 
X = features.values
y = train_xy['y'].values
train_y = train_xy.y

#correlation map
f,ax = plt.subplots(figsize=(40, 40))
sns.heatmap(features.corr(), annot=True, linewidths=.9, fmt= '.1f',ax=ax)


'''
feature selection
''' 

# method1 : filter using f-test
num_fea_sel = sklearn.feature_selection.f_regression(numerical_features, Y)
cat_fea_sel = sklearn.feature_selection.f_classif(categorical_features, Y)
    
# method2: removing features with low variance  
features_2 = VarianceThreshold(1).fit_transform(features)

# method3: univariate feature selection
#features_3 = SelectKBest(chi2, k=90).fit_transform(X, y)


# method4: L1-based feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
features_4 = model.transform(X)

# method5: Tree-based feature selection
clf_5 = ExtraTreesClassifier(n_estimators=90)
clf_5 = clf_5.fit(X, y)
clf_5.feature_importances_ 
model_5 = SelectFromModel(clf_5, prefit=True)
features_5 = model_5.transform(X)

# method6: wrapper: Feature ranking with recursive feature elimination.
features_6 = RFE(estimator=LogisticRegression(), n_features_to_select=100).fit_transform(X, y)

# method7: embedded: Logistic Regression based on L1 
features_7 = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(X, y)

# method8: embedded: GBDT
features_8 = SelectFromModel(GradientBoostingClassifier()).fit_transform(X, y)

# method9: feature extraction: dimension reduction using PCA
pca = PCA(n_components = 100)
train_nolabel = pca.fit_transform(train_nolabel)
test_all = pca.transform(test)
explained_variance = pca.explained_variance_ratio_  
features_9 = PCA(n_components=100).fit_transform(test, y)

# output features as csv
np.savetxt(os.path.join(out_dir, 'features_2.csv'), features_2, delimiter=",")
#np.savetxt(os.path.join(out_dir, 'features_3.csv'), features_3, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_4.csv'), features_4, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_5.csv'), features_5, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_6.csv'), features_6, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_7.csv'), features_7, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_8.csv'), features_8, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_9.csv'), features_9, delimiter=",")
#np.savetxt(os.path.join(out_dir, 'features_10.csv'), features_10, delimiter=",")

# double check if the index of the selected feature set is a subset of the original features
features2_indices = get_feature_indices(data_set = features, feature_set = features_2)


# method10: feature extraction: dimension reduction using LDA
#features_10 = LDA(n_components=30).fit_transform(X, y)

'''
rank
'''
model = XGBClassifier()
model.fit(features, y)

print(model.feature_importances_)

# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
#print((len(model.feature_importances_)))
	
# plot feature importance
my_plot_importance(booster=model, figsize=(15,20))
pyplot.show()


# get selected features
#selection = SelectFromModel(model, threshold=0.02, prefit=True)
#selected_features = selection.transform(features)

# train model based on selected features
##selection_model = XGBClassifier()
#selection_model.fit(selected_features, y)

# get selected test set
#select_x_test = selection.transform(test)

# evaluate new model
#y_pred = selection_model.predict(select_x_test)


# Fit model using each importance as a threshold
# thresholds = np.unique(np.sort(model.feature_importances_))
thresholds = [0.001]
for thresh in thresholds:
    if thresh > 0:
    	# select features using threshold
    	selection = SelectFromModel(model, threshold=thresh, prefit=True)
    	selected_features = selection.transform(features)
    	# train model
    	selection_model = XGBClassifier()
    	selection_model.fit(selected_features, y)
    	# eval model
    	select_X_test = selection.transform(x_test)
    	y_pred = selection_model.predict(select_X_test)
    	predictions = [round(value) for value in y_pred]
    	accuracy = accuracy_score(y_test, predictions)
    	print("Thresh=%.3f, n=%d, Accuracy: %.6f%%" % (thresh, selected_features.shape[1], accuracy*100.0-95))

'''
model
'''

# model1： decision tree 
dep = np.arange(1,9)
param_grid = {'max_depth': dep}
clf = tree.DecisionTreeClassifier()
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

clf_cv.fit(features_2, y)
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))     

dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))      

for i, k in enumerate(dep):
    clf = tree.DecisionTreeClassifier(max_depth=k)
    clf.fit(x_train, y_train)
    train_accuracy[i] = clf.score(x_train, y_train)
    test_accuracy[i] = clf.score(x_test, y_test)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()    

clf = tree.DecisionTreeClassifier(max_depth=1)
clf.fit(features_2, y)
y_predict1 = clf.predict(test)
np.savetxt(os.path.join(out_dir, 'y_predict1.csv'), y_predict1, delimiter=",")

#model2: LGBM
LGBM_model = LGBMClassifier(n_jobs=-1, learning_rate=0.01, min_split_gain=0.02, reg_alpha=0.04,
                            min_child_weight=0, reg_lambda=0.07, n_estimators=200, max_depth=4, num_leaves=20
                            )
LGBM_model.fit(selected_features, y)


select_x = selection.transform(test)
y_predict2 = LGBM_model.predict_proba(select_x)

np.savetxt(os.path.join(out_dir, 'y_predict2.csv'), y_predict2, delimiter=",")




#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_sample(X_train, Y)
#sorted(Counter(y_resampled).items())

