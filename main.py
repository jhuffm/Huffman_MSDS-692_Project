import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import uniform
#from sklearn import metrics
import matplotlib.pyplot as plt

training_set_features = pd.read_csv("https://s3.amazonaws.com/drivendata-prod/data/66/public/training_set_features.csv", index_col = 'respondent_id')
training_set_labels = pd.read_csv("https://s3.amazonaws.com/drivendata-prod/data/66/public/training_set_labels.csv", index_col = 'respondent_id')
test_set_features = pd.read_csv("https://s3.amazonaws.com/drivendata-prod/data/66/public/test_set_features.csv", index_col = 'respondent_id')

training_set_features.plot(kind = 'hist', subplots=True, layout=(6,6), figsize=(20,20))
plt.show()

##Exploratory Data Analysis

print(training_set_features.head())
print(training_set_labels.head())
print(test_set_features.head())

print(training_set_features.info())
print(training_set_features.describe())
print(training_set_features.isnull().sum())
training_set_features = training_set_features.drop(columns=['health_insurance', 'employment_industry', 'employment_occupation'])
print(training_set_features.isnull().sum())

##Encode categorical data
# Ordinal data: age_group, education, income_poverty
# Nominal data: race, sex, marital_status, hhs_geo_region, census_msa

# Using pandas factorize method for ordinal data -- factorizing column data
categories = pd.Categorical(training_set_features['age_group'], categories=['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'], ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['age_group'] = labels

categories = pd.Categorical(training_set_features['education'], categories=['< 12 Years', '12 Years', 'Some College', 'College Graduate'], ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['education'] = labels

categories = pd.Categorical(training_set_features['income_poverty'], categories=['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'], ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['income_poverty'] = labels

###Plot correlations in training set
joined_set = training_set_features.join(training_set_labels)
print(joined_set.head())
fig, ax = plt.subplots(figsize=(20,20))
corr = joined_set.corr()
htmp = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    ax=ax,
    annot=False
)
htmp.set_xticklabels(
    htmp.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

##One hot encode nominal categorical variables
training_set_features = pd.get_dummies(training_set_features, columns=['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa'], 
                                       prefix = ['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa'])

##Impute missing values using KNN
imputer = KNNImputer(n_neighbors=10)
training_set_features = imputer.fit_transform(training_set_features)
print(training_set_features.isnull().sum())

##Scale Data
scale = StandardScaler()
scaled_training_features =  scale.fit_transform(training_set_features)

###Build Prediction models
##Logistic Regression First

#Split into test and training set using 80% for training and 20% for testing so that we can keep test_set_labels data outside of the model
X_train, X_test, y_train, y_test = train_test_split(scaled_training_features, training_set_labels, test_size=0.2, random_state=0)

model1 = MultiOutputClassifier(LogisticRegression())  
model1.fit(X_train, y_train)
preds = model1.predict_proba(X_test)

model2 = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=300))
model2.fit(X_train, y_train)
preds2 = model2.predict_proba(X_test)

model3 = MultiOutputClassifier(RandomForestClassifier())
model3.fit(X_train, y_train)
preds3 = model3.predict_proba(X_test)

model4 = MultiOutputClassifier(SGDClassifier(loss="log"))
model4.fit(X_train, y_train)
preds4 = model4.predict_proba(X_test)

model5 = MultiOutputClassifier(SVC(probability = True))
model5.fit(X_train, y_train)
preds5 = model5.predict_proba(X_test)

model6 = MultiOutputClassifier(ExtraTreesClassifier())
model6.fit(X_train, y_train)
preds6 = model6.predict_proba(X_test)

def prediction(preds):
    y_preds = pd.DataFrame(
        {
            "h1n1_vaccine": preds[0][:,1],
            "seasonal_vaccine": preds[1][:,1],
        },
        index = y_test.index
    )
    return y_preds

y_preds = prediction(preds)
y_preds2 = prediction(preds2)
y_preds3 = prediction(preds3)
y_preds4 = prediction(preds4)
y_preds5 = prediction(preds5)
y_preds6 = prediction(preds6)

print(roc_auc_score(y_test, y_preds))
print(roc_auc_score(y_test, y_preds2))
print(roc_auc_score(y_test, y_preds3))
print(roc_auc_score(y_test, y_preds4))
print(roc_auc_score(y_test, y_preds5))
print(roc_auc_score(y_test, y_preds6))

### Parameter Optimization using GridSearchCV() and RandomizedSearchCV()
## Choosing logistic regression and random forest to optimize

parameter_grid = {'estimator__C': [0.001,0.01,0.1,1,10,100],
            'estimator__penalty' : ['l1', 'l2']}

grid_regres_class = GridSearchCV(
    estimator =  MultiOutputClassifier(LogisticRegression(solver='saga', max_iter=200)),
    param_grid = parameter_grid,
    scoring = 'roc_auc',
    n_jobs = 2,
    refit = True,
    cv = 5,
    return_train_score = True)

grid_regres_class.fit(X_train, y_train)

cv_results_df = pd.DataFrame(grid_regres_class.cv_results_)
best_row = cv_results_df[cv_results_df["rank_test_score"]==1]
print(best_row)

preds7 = grid_regres_class.predict_proba(X_test)
y_preds7 = prediction(preds7)
print(roc_auc_score(y_test, y_preds7))

## Random Forest parameter optimization
n_estimators = [10, 50, 100, 200, 400, 600, 800, 1000, 2000]
max_depth = [10, 20, 30, 40, 60, 80, 100, None]
min_samples_split = [2, 5, 10, 20, 50]
min_samples_leaf = [1, 2, 4]
max_features = ['auto', 'log2']
random_grid = {'estimator__n_estimators': n_estimators,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__max_features': max_features}


random_forest_class = RandomizedSearchCV(
    estimator = MultiOutputClassifier(RandomForestClassifier()),
    param_distributions = random_grid,
    n_iter = 10,
    scoring = 'roc_auc',
    n_jobs = 4,
    refit = True,
    cv = 5,
    random_state = 1,
    return_train_score = True)
random_forest_class.fit(X_train, y_train)

print(random_forest_class.best_params_)

preds8 = random_forest_class.predict_proba(X_test)
y_preds8 = prediction(preds8)
print(roc_auc_score(y_test, y_preds8))
