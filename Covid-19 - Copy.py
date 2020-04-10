
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:01:01 2020

@author: Priyanshi Chakrabort
"""

import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling

Covid_cases =pd.read_csv("cases.CSV")

Covid_cases.head()
print(Covid_cases.dtypes)


# Now we preprocess the first set of data 

print(Covid_cases.isnull().values.sum())

print(Covid_cases.isnull().sum())

Covid_cases = Covid_cases.fillna(Covid_cases['travel_history_country'].value_counts().index[0])
Covid_cases = Covid_cases.fillna(Covid_cases['locally_acquired'].value_counts().index[0])
Covid_cases = Covid_cases.fillna(Covid_cases['additional_info'].value_counts().index[0])
Covid_cases = Covid_cases.fillna(Covid_cases['additional_source'].value_counts().index[0])
Covid_cases = Covid_cases.fillna(Covid_cases['method_note'].value_counts().index[0])

print(Covid_cases.isnull().sum())

# removing Categorical Features using Label Encoder
Covid_cases_sklearn = Covid_cases.copy()
lb = LabelEncoder()
Covid_cases_sklearn['age'] = lb.fit_transform(Covid_cases['age'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['sex'] = lb.fit_transform(Covid_cases['sex'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['health_region'] = lb.fit_transform(Covid_cases['health_region'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['province'] = lb.fit_transform(Covid_cases['province'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['country'] = lb.fit_transform(Covid_cases['country'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['date_report'] = lb.fit_transform(Covid_cases['date_report'])

Covid_cases_sklearn.head()
Covid_cases_sklearn['report_week'] = lb.fit_transform(Covid_cases['report_week'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['travel_history_country'] = lb.fit_transform(Covid_cases['travel_history_country'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['locally_acquired'] = lb.fit_transform(Covid_cases['locally_acquired'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['case_source'] = lb.fit_transform(Covid_cases['case_source'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['additional_info'] = lb.fit_transform(Covid_cases['additional_info'])

Covid_cases_sklearn.head()

Covid_cases_sklearn['additional_source'] = lb.fit_transform(Covid_cases['additional_source'])

Covid_cases_sklearn.head()

Covid_mortality =pd.read_csv("mortality.CSV")
Covid_mortality.head()
print(Covid_mortality.dtypes)

print(Covid_mortality.isnull().values.sum())

print(Covid_mortality.isnull().sum())

Covid_mortality = Covid_mortality.fillna(Covid_mortality['case_id'].value_counts().index[0])
Covid_mortality = Covid_mortality.drop(['additional_info', 'additional_source'] , axis=1)

print(Covid_mortality.isnull().sum())

print(Covid_mortality.dtypes)


Covid_mortality.astype({'case_id':int})

print(Covid_mortality.dtypes)

#categorical Features
Covid_mortality_sklearn = Covid_mortality.copy()
lb1 = LabelEncoder()
Covid_mortality_sklearn['age'] = lb1.fit_transform(Covid_mortality['age'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['sex'] = lb1.fit_transform(Covid_mortality['sex'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['health_region'] = lb1.fit_transform(Covid_mortality['health_region'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['province'] = lb1.fit_transform(Covid_mortality['province'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['country'] = lb1.fit_transform(Covid_mortality['country'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['date_death_report'] = lb1.fit_transform(Covid_mortality['date_death_report'])

Covid_mortality_sklearn.head()

Covid_mortality_sklearn['death_source'] = lb1.fit_transform(Covid_mortality['death_source'])

Covid_mortality_sklearn.head()

#Merging the above Dataframes

Covid_1 = pd.merge(left=Covid_cases_sklearn, right=Covid_mortality_sklearn, left_on='case_id', right_on='case_id')
# In this case `case_id` is the only column name in  both dataframes, so if we skipped `left_on`
# And `right_on` arguments we would still get the same result

# size of the output data
Covid_1.shape
Covid_1

#Covid_1 contains both cases and mortality table data using inner join
Covid_recovered=pd.read_csv("recovered.CSV")
Covid_recovered.head()
print(Covid_recovered.dtypes)

Covid_testing= pd.read_csv("testing.CSV")
Covid_testing.head()
print(Covid_testing.dtypes)

Covid_2 = pd.merge(left=Covid_recovered, right=Covid_testing, left_on='province', right_on='province')
# In this case `case_id` is the only column name in  both dataframes, so if we skipped `left_on`
# And `right_on` arguments we would still get the same result

# size of the output data
Covid_2.shape
Covid_2

#Lets visualize the data that we have through a pie chart
import matplotlib.pyplot as plt

# Data to plot
labels = 'locally_acquired', 'travel_yn'
sizes = [215, 130]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

import seaborn as sns
Category_count = Covid_1['locally_acquired'].value_counts()
sns.set(style="darkgrid")
sns.barplot(Category_count.index, Category_count.values, alpha=0.9)
plt.title('Means of Contact on Local Basis')
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Reasons', fontsize=12)
plt.show()

print(Covid_2.isnull().values.sum())

print(Covid_2.isnull().sum())

Covid_2 = Covid_2.fillna(Covid_2['cumulative_recovered'].value_counts().index[0])
Covid_2 = Covid_2.fillna(Covid_2['province_source_x'].value_counts().index[0])
Covid_2 = Covid_2.fillna(Covid_2['source_x'].value_counts().index[0])
Covid_2 = Covid_2.fillna(Covid_2['cumulative_testing'].value_counts().index[0])
Covid_2 = Covid_2.fillna(Covid_2['province_source_y'].value_counts().index[0])
Covid_2 = Covid_2.fillna(Covid_2['source_y'].value_counts().index[0])

print(Covid_2.isnull().sum())

Covid_2 = Covid_2.drop(['province_source_x', 'source_x','province_source_y','source_y'] , axis=1)

print(Covid_2)



Covid_2_sklearn = Covid_2.copy()
lb2 = LabelEncoder()
Covid_2_sklearn['date_recovered'] = lb2.fit_transform(Covid_2['date_recovered'])
Covid_2_sklearn.head()

Covid_2_sklearn['province'] = lb2.fit_transform(Covid_2['province'])
Covid_2_sklearn.head()

Covid_2_sklearn['cumulative_recovered'] = lb2.fit_transform(Covid_2['cumulative_recovered'])
Covid_2_sklearn.head()

Covid_2_sklearn['date_testing'] = lb2.fit_transform(Covid_2['date_testing'])
Covid_2_sklearn.head()

Covid_2_sklearn['cumulative_testing'] = lb2.fit_transform(Covid_2['cumulative_testing'])
Covid_2_sklearn.head()
 

X1 = Covid_2_sklearn.iloc[:, :-1].values 
Y1 = Covid_2_sklearn.iloc[:, -1].values

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=0)

sc_X = StandardScaler()
X1_train = sc_X.fit_transform(X1_train)
X1_test = sc_X.transform(X1_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")
	
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")	

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")	

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")	

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")	

regressor = RandomForestRegressor(n_estimators=2000, random_state=0)
regressor.fit(X1_train, Y1_train)
y1_pred = regressor.predict(X1_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(Y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y1_test, y1_pred)))
print("********************************************")	

import matplotlib. pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
x, y = make_classification(n_samples=1000,
                           n_features=5,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
forest = ExtraTreesClassifier(n_estimators=2000,
                              random_state=0)

forest.fit(x, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(Covid_2_sklearn.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1,Covid_2_sklearn.shape[1]])
plt.show()


