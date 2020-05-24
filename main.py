import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
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

#print(training_set_features.dtypes) #redundant with below
print(training_set_features.info())
print(training_set_features.describe())
print(training_set_features.isnull().sum())
training_set_features = training_set_features.drop(columns=['health_insurance', 'employment_industry', 'employment_occupation'])
print(training_set_features.isnull().sum())

###Impute missing values using mode
cols = ["h1n1_concern", "h1n1_knowledge", "behavioral_antiviral_meds", "behavioral_avoidance",
        "behavioral_face_mask", "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
        "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal", "chronic_med_condition",
        "child_under_6_months", "health_worker", "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", 
        "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk",
        "opinion_seas_sick_from_vacc", "age_group", "education", "race", "sex", "income_poverty",
        "marital_status", "rent_or_own", "employment_status", "hhs_geo_region", "census_msa", 
        "household_adults", "household_children"]
training_set_features[cols]=training_set_features[cols].fillna(training_set_features.mode().iloc[0])
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