README

my_feature_list.pkl, my_dataset.pkl, and my_classifier.pkl contain selected features, dataset used, and classifier respectively. Updated in poi_id.py and classifiers.py, used in tester.py to evaluate classifier. 

poi_id.py contains final classifier 

classifiers.py contains intermediate work on project
	contains all tested classifiers as well as GridSearchCV tuning
	can also be used to verify precision/recall scores for each classifier
	as well as SelectKBest and PCA scores. Outputs to same pkl files as 
	poi_id.y

tester.py evaluates model passed through my_classifier.pkl