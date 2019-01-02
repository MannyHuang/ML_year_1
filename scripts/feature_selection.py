#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:45:12 2018

@author: Manny
"""

# Importing the libraries
import pandas as pd
import os
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
from sklearn.metrics import classification_report
#from imblearn.over_sampling import RandomOverSampler
from collections import Counter
#from imblearn.combine import SMOTEENN
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.lda import LDA

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
train_withlabel.replace({-99:np.nan}, inplace=True)
train_xy.replace({-99:np.nan}, inplace=True)
test.replace({-99:np.nan}, inplace=True)

'''
Analyze missing data
'''

# sort based on the ratio of missing data
sort_total_missing_data = train_withlabel.isnull().sum().sort_values(ascending=False)
sort_percent = (train_withlabel.isnull().sum()/train_withlabel.isnull().count()).sort_values(ascending = False)
sort_missing_data = pd.concat([sort_total_missing_data, sort_percent], axis = 1, keys = ['Total_missing_data', 'Percent'])

# Ploting the ratio of missing data
total_missing_data = train_withlabel.isnull().sum()
percent = train_withlabel.isnull().sum()/train_withlabel.isnull().count()
missing_data = pd.concat([total_missing_data, percent], axis = 1, keys = ['Total_missing_data', 'Percent'])
missing_data = missing_data['Percent'].values

'''
feature engineering
'''

# Dropping features with more than 99% missing values according to sort_missing_data
numerical_features = train_withlabel.iloc[:,0:96]
categorical_features = train_withlabel.iloc[:,95:]

numerical_features.drop(['x_94','x_92'],inplace = True,axis = 1)
categorical_features.drop(['x_129','x_132','x_134','x_116','x_110','x_112','x_118','x_113','x_126','x_131',
               'x_133','x_135','x_137','x_114','x_138','x_102','x_123','x_125','x_107','x_130',
               'x_119','x_128','x_115','x_109','x_117','x_127','x_103','x_111','x_108','x_136',
               'x_124','x_104','x_106','x_120','x_122','x_121','x_105'],inplace = True,axis = 1)   


'''
preprocessing
'''    

# impute missing numerical variables with mean
for i in numerical_features:
    numerical_features[i] = numerical_features[i].fillna(numerical_features[i].mean())   

 # Imputing missing caterogical variables
categorical_features = categorical_features.fillna(0)

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


'''
feature selection
''' 

# method1 : filter using f-test
num_fea_sel = sklearn.feature_selection.f_regression(numerical_features, Y)
cat_fea_sel = sklearn.feature_selection.f_classif(categorical_features, Y)
    
# method2: removing features with low variance  
features_2 = VarianceThreshold(threshold=3).fit_transform(features)

# method3: univariate feature selection
features_3 = SelectKBest(chi2, k=30).fit_transform(X, y)


# method4: L1-based feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
features_4 = model.transform(X)

# plot features
# plot_feature(test=test, train_withlabel=train_withlabel)

# output features as csv
np.savetxt(os.path.join(out_dir, 'features_2.csv'), features_2, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_3.csv'), features_3, delimiter=",")
np.savetxt(os.path.join(out_dir, 'features_4.csv'), features_4, delimiter=",")


"""
# 1.5 Tree-based feature selection
clf_5 = ExtraTreesClassifier(n_estimators=50)
clf_5 = clf_5.fit(X, y)
clf_5.feature_importances_ 
model_5 = SelectFromModel(clf_5, prefit=True)
feature_5 = model_5.transform(X)
"""

'''
# 1.6 Feature selection as part of a pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
'''

"""
# 2. Wrapper
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(X, y)


# 3. Embedded
# based on L1
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(X, y)

# based on L1 and L2
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(X, y)

# GBDT
SelectFromModel(GradientBoostingClassifier()).fit_transform(X, y)


# 4. Lowering the dimension
# 4.1 Applying PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


# 4.2 Applying LDA
LDA(n_components=None).fit_transform(X, y)



'''
## Decision tree classifier

train_set = features.iloc[:15000]
validation_set = features.iloc[:15000]

x = train_set.values
test = test.values
y = y_train.values


# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth': dep}

# Instantiate a decision tree classifier 
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

clf_cv.fit(x, y)
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))     



# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    clf = tree.DecisionTreeClassifier(max_depth=k)
    clf.fit(x_train, y_train)
    train_accuracy[i] = clf.score(x_train, y_train)
    test_accuracy[i] = clf.score(x_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()    

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x, y)
#y_predict = clf.predict(test)
'''


# over-sampling
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_sample(X_train, Y)

#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_sample(X_train, Y)
#sorted(Counter(y_resampled).items())

# logisticregression
#clf = Lo
"""