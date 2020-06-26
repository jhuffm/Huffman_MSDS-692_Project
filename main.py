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
from sklearn.metrics import multilabel_confusion_matrix
from scipy.stats import uniform
from skopt import BayesSearchCV
import matplotlib.pyplot as plt

training_set_features = pd.read_csv("https://raw.githubusercontent.com/jhuffm/Huffman_MSDS-692_Project/master/data/training_set_features.csv", index_col = 'respondent_id')
training_set_labels = pd.read_csv("https://raw.githubusercontent.com/jhuffm/Huffman_MSDS-692_Project/master/data/training_set_labels.csv", index_col = 'respondent_id')
test_set_features = pd.read_csv("https://raw.githubusercontent.com/jhuffm/Huffman_MSDS-692_Project/master/data/test_set_features.csv", index_col = 'respondent_id')
submission_format = pd.read_csv("https://raw.githubusercontent.com/jhuffm/Huffman_MSDS-692_Project/master/data/submission_format.csv", index_col = 'respondent_id')

##Exploratory Data Analysis
print(training_set_features.head())
print(training_set_labels.head())
print(test_set_features.head())

print(training_set_features.info())
print(training_set_features.describe())

print(training_set_labels['h1n1_vaccine'].value_counts())
print(training_set_labels['seasonal_vaccine'].value_counts())

training_set_features.plot(kind = 'hist', subplots=True, layout=(5,5), figsize=(20,20), sharey=True, title = 'Frequency Plots of Categorical Variables')
plt.show()

##Visualizations
joined_set = training_set_features.join(training_set_labels)
joined_set['h1n1_vaccine'] = joined_set['h1n1_vaccine'].replace(0, 'Not Vaccinated')
joined_set['h1n1_vaccine'] = joined_set['h1n1_vaccine'].replace(1, 'Vaccinated')
joined_set['seasonal_vaccine'] = joined_set['seasonal_vaccine'].replace(0, 'Not Vaccinated')
joined_set['seasonal_vaccine'] = joined_set['seasonal_vaccine'].replace(1, 'Vaccinated')
order_age = ['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years']
order_education = ['< 12 Years', '12 Years', 'Some College', 'College Graduate']
order_poverty = ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000']
f, axes = plt.subplots(6,2, sharex = False, sharey=False, figsize = (7,10))

sns.countplot( x = joined_set['opinion_seas_risk'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[0,0]).set_title('H1N1 Vaccine')
sns.countplot( x = joined_set['opinion_seas_risk'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[0,1]).set_title('Seasonal Vaccine')

sns.countplot( x = joined_set['opinion_h1n1_risk'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[1,0]).legend_.remove()
sns.countplot( x = joined_set['opinion_h1n1_risk'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[1,1]).legend_.remove()

sns.countplot( x = joined_set['education'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[2,0], order = order_education).set_xticklabels(order_education, rotation=12)
sns.countplot( x = joined_set['education'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[2,0], order = order_education).legend_.remove()
sns.countplot( x = joined_set['education'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[2,1], order = order_education).set_xticklabels(order_education, rotation = 12)
sns.countplot( x = joined_set['education'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[2,1], order = order_education).legend_.remove()

sns.countplot( x = joined_set['age_group'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[3, 0], order = order_age).set_xticklabels(order_age, rotation=12)
sns.countplot( x = joined_set['age_group'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[3, 0], order = order_age).legend_.remove()
sns.countplot( x = joined_set['age_group'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[3,1], order = order_age).set_xticklabels(order_age, rotation=12)
sns.countplot( x = joined_set['age_group'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[3,1], order = order_age).legend_.remove()

sns.countplot( x = joined_set['behavioral_face_mask'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[4,0]).legend_.remove()
sns.countplot( x = joined_set['behavioral_face_mask'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[4,1]).legend_.remove()

sns.countplot( x = joined_set['behavioral_wash_hands'], hue =joined_set['h1n1_vaccine'], data=joined_set, ax = axes[5,0]).legend_.remove()
sns.countplot( x = joined_set['behavioral_wash_hands'], hue =joined_set['seasonal_vaccine'], data=joined_set, ax = axes[5,1]).legend_.remove()

f.tight_layout(pad = 0.5)
plt.show()

##Encode categorical data
# Ordinal data: age_group, education, income_poverty
# Nominal data: race, sex, marital_status, hhs_geo_region, census_msa

print(training_set_features.isnull().sum())
training_set_features = training_set_features.drop(columns=['health_insurance', 'employment_industry', 'employment_occupation'])
test_set_features = test_set_features.drop(columns=['health_insurance', 'employment_industry', 'employment_occupation'])

cols = ["h1n1_concern", "h1n1_knowledge", "behavioral_antiviral_meds", "behavioral_avoidance",
        "behavioral_face_mask", "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
        "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal", "chronic_med_condition",
        "child_under_6_months", "health_worker", "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", 
        "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk",
        "opinion_seas_sick_from_vacc", "age_group", "education", "race", "sex", "income_poverty",
        "marital_status", "rent_or_own", "employment_status", "hhs_geo_region", "census_msa", 
        "household_adults", "household_children"]

training_set_features[cols]=training_set_features[cols].fillna(training_set_features.mode().iloc[0])
test_set_features[cols]=test_set_features[cols].fillna(test_set_features.mode().iloc[0])
print(training_set_features.isnull().sum())

# Using pandas factorize method for ordinal data -- factorizing column data
categories = pd.Categorical(training_set_features['age_group'], categories=order_age, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['age_group'] = labels
categories = pd.Categorical(test_set_features['age_group'], categories=order_age, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
test_set_features['age_group'] = labels

categories = pd.Categorical(training_set_features['education'], categories=order_education, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['education'] = labels
categories = pd.Categorical(test_set_features['education'], categories=order_education, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
test_set_features['education'] = labels

categories = pd.Categorical(training_set_features['income_poverty'], categories=order_poverty, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
training_set_features['income_poverty'] = labels
categories = pd.Categorical(test_set_features['income_poverty'], categories=order_poverty, ordered=True)
labels, unique = pd.factorize(categories, sort=True)
test_set_features['income_poverty'] = labels

###Plot correlations in training set to see if any variables are highly correlated
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
test_set_features = pd.get_dummies(test_set_features, columns=['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa'], 
                                       prefix = ['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa'])

##Scale Data
scale = StandardScaler()
scaled_training_features =  scale.fit_transform(training_set_features)
scaled_test_features =  scale.fit_transform(test_set_features)

###Build Prediction models

#Split into test and training set using 80% for training and 20% for testing so that we can keep test_set_labels data outside of the model
X_train, X_test, y_train, y_test = train_test_split(scaled_training_features, training_set_labels, test_size=0.2, random_state=0)

def fit_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    y_preds = pd.DataFrame(
        {
            "h1n1_vaccine": preds[0][:,1],
            "seasonal_vaccine": preds[1][:,1],
        },
        index = y_test.index
    )
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
    model_name = namestr(model, globals())
    print('The ' + str(model_name)[2:-2] + ' ROC AUC score is ' + str(roc_auc_score(y_test, y_preds)))

Logistic_regression = MultiOutputClassifier(LogisticRegression())  
fit_model(Logistic_regression, X_train, y_train, X_test)

KNN_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=300))
fit_model(KNN_model, X_train, y_train, X_test)

Random_forest = MultiOutputClassifier(RandomForestClassifier())
fit_model(Random_forest, X_train, y_train, X_test)

SGD_classifier = MultiOutputClassifier(SGDClassifier(loss="log"))
fit_model(SGD_classifier, X_train, y_train, X_test)

SVC_model = MultiOutputClassifier(SVC(probability = True))
fit_model(SVC_model, X_train, y_train, X_test)

Extra_trees_classifier = MultiOutputClassifier(ExtraTreesClassifier())
fit_model(Extra_trees_classifier, X_train, y_train, X_test)

### Parameter Optimization using GridSearchCV() and BayesSearchCV()
## Choosing logistic regression and random forest to optimize
parameter_grid = {'estimator__C': [0.001,0.01,0.1,1,10,100],
            'estimator__penalty' : ['l1', 'l2']}

Log_regression_grid_optimized_classifier = GridSearchCV(
    estimator =  MultiOutputClassifier(LogisticRegression(solver='saga', max_iter=200)),
    param_grid = parameter_grid,
    scoring = 'roc_auc',
    n_jobs = 2,
    refit = True,
    cv = 5,
    return_train_score = True)

fit_model(Log_regression_grid_optimized_classifier, X_train, y_train, X_test)
cv_results_df = pd.DataFrame(Log_regression_grid_optimized_classifier.cv_results_)
best_row = cv_results_df[cv_results_df["rank_test_score"]==1]
print(best_row)

## Random Forest parameter optimization w/ Bayesian Method
n_estimators = [10, 50, 100, 200, 400, 600, 800, 1000, 2000]
max_depth = [10, 20, 30, 40, 60, 80, 100, 1000]
min_samples_split = [2, 5, 10, 20, 50]
min_samples_leaf = [1, 2, 4]
max_features = ['auto', 'log2']
param_grid = {'estimator__n_estimators': n_estimators,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__max_features': max_features}

random_forest_Bayes_optimized_classifier = BayesSearchCV(
    MultiOutputClassifier(RandomForestClassifier()),
    param_grid,
    n_iter = 100,
    scoring = 'roc_auc',
    n_jobs = 4,
    refit = True,
    cv = 3,
    random_state = 1,
    return_train_score = True)

fit_model(random_forest_Bayes_optimized_classifier, X_train, y_train, X_test)
print(random_forest_Bayes_optimized_classifier.best_estimator_)

#Show Confusion Matrix
random_forest_optim = MultiOutputClassifier(RandomForestClassifier(n_estimators = 2000, max_depth = 20, min_samples_split = 20, min_samples_leaf = 4, max_features= 'auto')) 
classifier = random_forest_optim.fit(X_train, y_train)
cm = multilabel_confusion_matrix(y_test, random_forest_optim.predict(X_test))
print(cm)

## Retrain best model on full dataset and fit to test_set_features
random_forest_optim.fit(scaled_training_features, training_set_labels)
preds = random_forest_optim.predict_proba(scaled_test_features)

## Format for submittal on DrivenData
#Code copied from DrivenData to ensure correct format for submittal

# Save predictions to submission data frame
submission_format["h1n1_vaccine"] = preds[0][:, 1]
submission_format["seasonal_vaccine"] = preds[1][:, 1]

print(submission_format.head())
submission_format.to_csv('my_submission.csv', index= True)
