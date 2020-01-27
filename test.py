import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# to convert the tsv dataset in a csv file
# tsv_file='name.tsv'
# csv_table=pd.read_table(tsv_file,sep='\t')
# csv_table.to_csv('new_name.csv',index=False)

datasetTrain = pd.read_csv('.../drugsComTrain_raw.csv', header=0, sep=',')
datasetTest = pd.read_csv('.../drugsComTest_raw.csv', header=0, sep=',')

# deriving the three level polarity lables
for r in range(datasetTrain.shape[0]):
    if datasetTrain.iat[r, 4] <= 4:
        datasetTrain.iat[r, 4] = -1
    elif datasetTrain.iat[r, 4] < 7:
        datasetTrain.iat[r, 4] = 0
    else:
        datasetTrain.iat[r, 4] = 1

for r in range(datasetTest.shape[0]):
    if datasetTest.iat[r, 4] <= 4:
        datasetTest.iat[r, 4] = -1
    elif datasetTest.iat[r, 4] < 7:
        datasetTest.iat[r, 4] = 0
    else:
        datasetTest.iat[r, 4] = 1

reviews = datasetTrain.iloc[:, 3]
ratings = datasetTrain.iloc[:, 4]
test = datasetTest.iloc[:, 3]
target = datasetTest.iloc[:, 4]

# the hyper-parameters are found using gridSearchCV
text_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf', Perceptron()),
])

# Searching for the best parameters
# parameters = {
#  'vect__max_df': (0.5, 0.75, 1.0),
#  'vect__max_features': (None, 5000, 10000, 50000),
#  'vect__ngram_range': ((1, 2), (1, 3)),
#  'tfidf__use_idf': (True, False),
#  'tfidf__norm': ('l1', 'l2'),
#  'clf__alpha': (0.00001, 0.000001),
#  'clf__penalty': ('l2', None, 'elasticnet’')
#  }


# grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1, verbose=1, cv=5)
# grid_search.fit(reviews, ratings)
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#The default parameters were already the best ones, with the exception of including word bi-gram and tri-gram features

text_clf.fit(reviews, ratings)

predicted = text_clf.predict(test)

print('Accuracy:  %0.4f' % accuracy_score(target, predicted))

