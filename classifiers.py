#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

features_list = ['poi', 'salary', 'salary_bonus_ratio', 'total_minus_exercised', 
				'to_messages', 'deferral_payments', 'total_payments', 
				'exercised_stock_options', 'bonus', 'restricted_stock', 
				'shared_receipt_with_poi', 'restricted_stock_deferred', 
				'total_stock_value', 'expenses', 'loan_advances', 
				'from_messages', 'other', 'from_this_person_to_poi', 
				'director_fees', 'deferred_income', 'long_term_incentive', 
				'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop("TOTAL", None)

### Create new features
#Salary/bonus ratio
for name in data_dict:
	if (data_dict[name]['salary'] != 'NaN') and \
			(data_dict[name]['bonus'] != 'NaN'):
		data_dict[name].update({'salary_bonus_ratio' : \
			(float(data_dict[name]['salary']) / data_dict[name]['bonus'])})
	else:
		data_dict[name].update({'salary_bonus_ratio' : 'NaN'})

#total stock options - exercised stock options
for name in data_dict:
	if (data_dict[name]['exercised_stock_options'] != 'NaN') and \
			(data_dict[name]['total_stock_value'] != 'NaN'):
		data_dict[name].update({'total_minus_exercised' : \
			(float(data_dict[name]['total_stock_value']) - \
			data_dict[name]['exercised_stock_options'])})
	else:
		data_dict[name].update({'total_minus_exercised' : 'NaN'})


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

### 6 possible feature selection/reduction and classifier combinations
### Uncomment estimator and parameter for one and run to get optimal parameters.
### Optimal parameters may be added manually to estimators for testing purposes
### with tester.py 

#Naive Bayes w/SelectKBest
'''
estimators = [('feature_selection', SelectKBest()), 
				('clf', GaussianNB())]
parameters = {'feature_selection__k' : (1, 2, 3, 4, 5, 6, 7, 8)}
'''
#Naive Bayes w/PCA
'''
estimators = [('reduce_dim', PCA()), ('clf', GaussianNB())]
parameters = {'reduce_dim__n_components' : (1, 2, 3, 4, 5, 6, 7, 8)}
'''
#Decision Tree w/SelectKBest
'''
estimators = [('feature_selection', SelectKBest()), 
				('clf', tree.DecisionTreeClassifier())]
parameters = {'feature_selection__k' : (1, 2, 3, 4, 5, 6, 7, 8),
				'clf__min_samples_split' : (2, 10, 50, 100, 125)}
'''
#Decision Tree w/PCA
'''
estimators = [('reduce_dim', PCA()), ('clf', tree.DecisionTreeClassifier())]
parameters = {'reduce_dim__n_components' : (1, 2, 3, 4, 5, 6, 7, 8),
				'clf__min_samples_split' : (2, 10, 50, 100, 125)}
'''
#Adaboost w/SelectKBest
'''
estimators = [('feature_selection', SelectKBest()), 
				('clf', AdaBoostClassifier())]
parameters = {'feature_selection__k' : (1, 2, 3, 4, 5, 6, 7, 8),
				'clf__learning_rate' : (.01, .05, .1, .5, 1, 1.5, 2),
				'clf__n_estimators' : (1, 3, 5, 7, 9, 11)}
'''
#Adaboost w/PCA
'''
estimators = [('reduce_dim', PCA()), ('clf', AdaBoostClassifier())]
parameters = {'reduce_dim__n_components' : (1, 2, 3, 4, 5, 6, 7, 8),
				'clf__learning_rate' : (.01, .05, .1, .5, 1, 1.5, 2),
				'clf__n_estimators' : (1, 3, 5, 7, 9, 11)}
'''
clf = Pipeline(estimators)

#Runs a GridSearch to get optimal parameters 
optimizer = GridSearchCV(clf, parameters)
optimizer.fit(features, labels)
#prints optimal parameters
print("Best parameters set:")
best_parameters = optimizer.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

clf.fit(features, labels)

#Uncomment to get scores, pvalues from models with KBest feature selection
#Optimal parameters must be added to estimators to get correct scores
'''
KBestScores = clf.named_steps['feature_selection'].scores_
KBestPvalues = clf.named_steps['feature_selection'].pvalues_
chosen_features = clf.named_steps['feature_selection'].get_support()
mod_features_list = features_list[1:]
for (chosen_features, mod_features_list, scores, pvalues) in \
		zip(chosen_features, mod_features_list, KBestScores, KBestPvalues):
	if chosen_features == True:
		print mod_features_list, scores, pvalues
'''
#Uncomment to get scores with models with PCA feature reduction
#As above, optimal parameters must be added
'''
pca_score = clf.named_steps['reduce_dim'].explained_variance_ratio_
print pca_score
'''


dump_classifier_and_data(clf, my_dataset, features_list)