
# coding: utf-8

# In[187]:

import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


scalar = StandardScaler()

#Loading dataset

df = pd.read_csv('Seasons_Stats.csv',encoding = "ISO-8859-1")
print(df.shape)
print(df[(df.firstTeam == 1)])


#Shaping dataframes
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='any')

df.drop(['Player'], axis = 1, inplace = True)
df.drop(['Pos'], axis = 1, inplace = True)
df.drop(['Tm'], axis = 1, inplace = True)


firstTeam_df = df[(df.firstTeam == 1) & (df.Age.notnull())]

noTeam_df = df[(df.firstTeam == 0) & (df.Age.notnull())]

print(firstTeam_df.shape)
print(noTeam_df.shape)

first_df = firstTeam_df.iloc[:,:-1]
no_df = noTeam_df.iloc[:,:-1]
features = pd.concat([first_df, no_df])

labelsfirst = firstTeam_df.iloc[:, -1]
labelsno = noTeam_df.iloc[:, -1]
labels = pd.concat([labelsfirst, labelsno])


# In[168]:

#Running classifiers on dataframes
from sklearn import tree
NBADTree = tree.DecisionTreeClassifier()

NBADTree.fit(features,labels)#look at confusion matrix
print("The Decision Trees accuracy",NBADTree.score(features,labels))
#print(np.around(features.values,1))
#print((np.around(NBANeighb.predict_proba(features),decimals=1)))
print("The Decision Trees precision",precision_score(labels.values,(NBADTree.predict(features))))
NBABayes = GaussianNB()
#NBABayes = KNeighborsClassifier()

NBABayes.fit(features,labels)#look at confusion matrix
print("The Naive Bayes accuracy",NBABayes.score(features,labels))
#print(np.around(features.values,1))
#print((np.around(NBABayes.predict_proba(features),decimals=1)))
print("The Naive Bayes precision",precision_score(labels.values,(NBABayes.predict(features))))

NBANeighb = KNeighborsClassifier()

NBANeighb.fit(features,labels)#look at confusion matrix
print("The KNN accuracy",NBANeighb.score(features,labels))
#print(np.around(features.values,1))
#print((np.around(NBANeighb.predict_proba(features),decimals=1)))
print("The KNN precision",precision_score(labels.values,(NBANeighb.predict(features))))

NBALog = LogisticRegression()

NBALog.fit(features,labels)#look at confusion matrix
print("The Logistic Regression accuracy",NBALog.score(features,labels))
#print(np.around(features.values,1))
#print((np.around(NBANeighb.predict_proba(features),decimals=1)))
print("The Logistic Regression precision",precision_score(labels.values,(NBALog.predict(features))))


# In[206]:

#Adopted from sklearn doc script: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#Using sklearn script, displaying feature importances using forest
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

NBAForest = ExtraTreesClassifier(n_estimators=500,
                              random_state=0)

NBAForest.fit(features, labels)
print("The Forest accuracy",NBAForest.score(features,labels))

print("The Decision Trees precision",precision_score(labels.values,(NBADTree.predict(features))))
importances = NBAForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(features.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #print(indices[f])

# Plot the feature importances of the forest
plt.figure(figsize=(18,8))
plt.title("Feature importances")
plt.bar(range(features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
plt.show()


# In[171]:

for i in range(len(features.columns.values)):
    print(i,features.columns.values[i])
print(features.shape)


# In[180]:

training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size=0.15, random_state=22)


kfoldforest = ExtraTreesClassifier(n_estimators=500,random_state=0)
kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[177]:
predictions = NBABayes.predict(testing_features)

#precision using 10 fold cross validation
scores = cross_val_score(NBABayes, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score for Naive Bayes is " + str(scores.mean()))


# In[178]:

predictions = NBANeighb.predict(testing_features)

#precision using 10 fold cross validation
scores = cross_val_score(NBANeighb, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score for KNN is " + str(scores.mean()))


# In[181]:

predictions = NBALog.predict(testing_features)

#precision using 10 fold cross validation
scores = cross_val_score(NBALog, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score for Logistic Regression is " + str(scores.mean()))


# In[189]:
#Experimenting with feature selection
top10feat = features.drop(['Year','Age','G','GS','MP','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OBPM','DBPM','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF'], axis=1, inplace=True)


print(top10feat)


# In[190]:

training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size=0.15, random_state=22)


kfoldforest = ExtraTreesClassifier(n_estimators=500,random_state=0)

kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))
print(features)


# So with just our top 10 features, we get a very similar precision at 0.8133.

# In[191]:

import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


scalar = StandardScaler()

print('Loading training data...')

df = pd.read_csv('Seasons_Stats.csv',encoding = "ISO-8859-1")
print(df.shape)
print(df[(df.firstTeam == 1)])


df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='any')
#df = df.drop(["Player","Pos","Tm"])
df.drop(['Player'], axis = 1, inplace = True)
df.drop(['Pos'], axis = 1, inplace = True)
df.drop(['Tm'], axis = 1, inplace = True)


firstTeam_df = df[(df.firstTeam == 1) & (df.Age.notnull())]

noTeam_df = df[(df.firstTeam == 0) & (df.Age.notnull())]


print(firstTeam_df.shape)
print(noTeam_df.shape)

first_df = firstTeam_df.iloc[:,:-1]
no_df = noTeam_df.iloc[:,:-1]
features = pd.concat([first_df, no_df])

labelsfirst = firstTeam_df.iloc[:, -1]
labelsno = noTeam_df.iloc[:, -1]
labels = pd.concat([labelsfirst, labelsno])


# In[195]:

training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size=0.15, random_state=22)

#Experimenting with n_estimators (number of trees in forest) hyperparameter
kfoldforest = ExtraTreesClassifier(n_estimators=5,random_state=22)

kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[197]:

training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size=0.15, random_state=22)

kfoldforest = ExtraTreesClassifier(n_estimators=10,random_state=22)

kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[200]:

kfoldforest = ExtraTreesClassifier(n_estimators=50,random_state=22)

kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[201]:

kfoldforest = ExtraTreesClassifier(n_estimators=100,random_state=22)
kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[202]:

kfoldforest = ExtraTreesClassifier(n_estimators=250,random_state=22)
kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[203]:

kfoldforest = ExtraTreesClassifier(n_estimators=500,random_state=22)
kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[204]:

kfoldforest = ExtraTreesClassifier(n_estimators=1000,random_state=22)
kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))


# In[205]:

kfoldforest = ExtraTreesClassifier(n_estimators=10000,random_state=22)

kfoldforest.fit(training_features, training_labels)

predictions = kfoldforest.predict(testing_features)

scores = cross_val_score(kfoldforest, training_features, training_labels, cv=10, scoring='precision')
print (scores)
print ("Average score is " + str(scores.mean()))
