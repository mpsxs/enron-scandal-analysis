#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Feature selection
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

### Remove "TOTAL" outlier
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

### Build classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

estimators = [('feature_selection', SelectKBest(f_regression, k=5)), 
				('clf', GaussianNB())]
clf = Pipeline(estimators)

dump_classifier_and_data(clf, my_dataset, features_list)