#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
import pickle
from sklearn.ensemble import RandomForestClassifier


# set the aesthetic style of the plots
sns.set_style()

# filter warning messages
import warnings


# In[11]:


# set default matplotlib parameters
COLOR = '#ababab'
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['grid.color'] = COLOR
mpl.rcParams['grid.alpha'] = 0.1


# In[12]:


#read file
fname='credit_info.xlsx'
file1=os.path.join(os.getcwd(),fname)


# In[13]:


#read the sheets with different dataframe
df1 = pd.read_excel(file1, 'Person')
df2 = pd.read_excel(file1, 'Loan')

#merge both the dataframe with common column name "Person_ID"

df_credit=pd.merge(df1,df2,on='Person_ID')


# In[14]:


df_credit.head()


# In[16]:


#Data Analysis


# In[17]:


# data frame shape
print('Number of rows: ', df_credit.shape[0])
print('Number of columns: ', df_credit.shape[1])


# In[18]:


# We are working with a data set containing 28 features for 2593 clients. Bad Indicator is a 0/1 feature and is the target variable we are trying to predict. We'll explore all features searching for outliers, treating possible missing values, and making other necessary adjustments to improve the overall quality of the model.

# data frame summary
df_credit.info()


# In[19]:


# percentage of missing values per feature
print((df_credit.isnull().sum() * 100 / df_credit.shape[0]).sort_values(ascending=False))


# In[20]:


#First of all, note that Bad Indicator has missing values. As this is our target variable, we don't have a lot of options here. So, we'll eliminate all entries where Bad Indicator is null.

df_credit.dropna(subset=['Bad Indicator'], inplace=True)


# In[21]:


#Observe that BUSINESS_TYPE has almost 50% its entries missing. As this feature is not crucial for the project, we are dropping it.

# drop the column "BUSINESS_TYPE"
df_credit.drop('BUSINESS_TYPE', axis=1, inplace=True)


# In[22]:


#  Now , let's examine the number of unique values for each feature.'

# number of unique observations per column
df_credit.nunique().sort_values()


# In[23]:


#The features BASIC_SAVINGS  and Payment Period have only one value. As that won't be useful for the model, we can drop these two columns.

# drop the columns "BASIC_SAVINGS" and "Payment Period"
df_credit.drop(labels=['BASIC_SAVINGS', 'Payment Period'], axis=1, inplace=True)


# In[24]:


# Moving on with the cleaning process, to keep the data set as adequate as possible we'll remove some other columns that are not adding value to the model. Some features, as score_1 and score_2, are filled with hashed values. However, we are keeping these variables as they might be useful to our model.

df_credit.drop(labels=['Relationship_Start_Date', 'No_of_Mobile_No', 'DATE_OF_BIRTH', 'OCCUPATION', 'REGION', 'Loan_ID_y',
                       'Loan Application Date', 'Loan Approval Date', 'Loan Disbursement Date', 'Loan Maturity Date'], axis=1, inplace=True)


# In[25]:


#now we are working with a leaner data set. Before dealing with the missing values, let's examine if there are outliers in the data set. We'll start by taking a look at some statistical details of the numerical feature


# show descriptive statistics
df_credit.describe()


# In[26]:


# new data frame, containing numerical features of interest, will be created. Plotting a histogram for these features will helps us examine their distribution.

# data frame containing numerical features
df_credit_numerical = df_credit[['Ever 90dpd+', 'Currently ≥ 60dpd']]


# In[27]:


# plot a histogram for each of the features above 

nrows = 2
ncols = 2

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 16))

r = 0
c = 0

for i in df_credit_numerical:
  sns.distplot(df_credit_numerical[i], bins=15,kde=False, ax=ax[r][c])
  if c == ncols - 1:
    r += 1
    c = 0
  else:
    c += 1

plt.show()


# In[28]:


#All these features above have missing values that need to be treated. As we can see, they have skewed distribution, which is an indication that we should fill the missing values with the median value for each feature.

#It's time to deal with the missing values from the remaining  columns. We are filling these values according to the particularities of each feature, as below:

#Categorical variables will be filled with the most recurrent value.
#Numerical variables will be filled with their median values.


# In[29]:


df_credit_num = df_credit.select_dtypes(exclude='object').columns
df_credit_cat = df_credit.select_dtypes(include='object').columns


# In[30]:


df_credit_num


# In[31]:


df_credit_cat


# In[32]:


df_credit.drop(labels=['CUSTOMER_EMAIL'],axis=1)


# In[33]:



# fill missing values for numerical variables
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(df_credit.loc[:, df_credit_num])
df_credit.loc[:, df_credit_num] = imputer.transform(df_credit.loc[:, df_credit_num])

# fill missing values for categorical variables
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(df_credit.loc[:, df_credit_cat])
df_credit.loc[:, df_credit_cat] = imputer.transform(df_credit.loc[:, df_credit_cat])


# In[34]:


df_credit.isnull().sum()


# In[35]:


#After handling the missing values, case by case, we now have a data set free of null values.

#We'll now preprocess the data, converting the categorical features into numerical values. LabelEncoder will be used for the binary variables while get_dummies will be used for the other categorical variables.

bin_var = df_credit.nunique()[df_credit.nunique() == 2].keys().tolist()
num_var = [col for col in df_credit.select_dtypes(['int', 'float']).columns.tolist() if col not in bin_var]
cat_var = [col for col in df_credit.select_dtypes(['object']).columns.tolist() if col not in bin_var]
df_credit_encoded = df_credit.copy()

# label encoding for the binary variables
le = LabelEncoder()
for col in bin_var:
  df_credit_encoded[col] = le.fit_transform(df_credit_encoded[col])

# encoding with get_dummies for the categorical variables
df_credit_encoded = pd.get_dummies(df_credit_encoded, columns=cat_var)

df_credit_encoded.head()


# In[36]:


#After encoding the categorical variables, let's start working on the machine learning models.

#Machine Learning Models
#We are experimenting with the following 3 boosting algorithms to determine which one yields better results:

#XGBoost
#LightGBM
#CatBoost
#Before starting with the models, let's split the data into training and test sets.


# In[37]:


# feature matrix
X = df_credit_encoded.drop('Bad Indicator', axis=1)

# target vector
y = df_credit_encoded['Bad Indicator']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)


# In[38]:


from imblearn.under_sampling import RandomUnderSampler


# In[39]:


#Now, as we are dealing with an unbalanced data set, we'll standardize and resample the training set, with StandardScaler and RandomUnderSampler, respectively.

# standardize numerical variables
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# resample
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


# In[40]:


# define the function val_model
def val_model(X, y, clf, show=True):
    """
    Apply cross-validation on the training set.

    # Arguments
        X: DataFrame containing the independent variables.
        y: Series containing the target vector.
        clf: Scikit-learn estimator instance.
        
    # Returns
        float, mean value of the cross-validation scores.
    """
    
    X = np.array(X)
    y = np.array(y)

    pipeline = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipeline, X, y, scoring='recall')

    if show == True:
        print(f'Recall: {scores.mean()}, {scores.std()}')
    
    return scores.mean()


# In[41]:


#evaluate the models
xgb = XGBClassifier()
lgb = LGBMClassifier()
cb = CatBoostClassifier()

model = []
recall = []

for clf in (xgb, lgb, cb):
    model.append(clf.__class__.__name__)
    recall.append(val_model(X_train_rus, y_train_rus, clf, show=False))

pd.DataFrame(data=recall, index=model, columns=['Recall'])


# In[42]:


#XGBoost

#XGBoost is known for being one of the most effective Machine Learning algorithms, due to its good performance on structured and tabular datasets on classification and regression predictive modeling problems. It is highly customizable and counts with a large range of hyperparameters to be tuned.

##For the XGBoost model, we'll tune the following hyperparameters:

#n_estimators - The number of trees in the model
#max_depth - Maximum depth of a tree
#min_child_weight - Minimum sum of instance weight needed in a child
#gamma - Minimum loss reduction required to make a further partition on a leaf node of the tree
#learning_rate - Step size shrinkage used in the update to prevents overfitting


# In[43]:


# XGBoost
xgb = XGBClassifier()

# parameter to be searched
param_grid = {'n_estimators': range(0,1000,50)}

# find the best parameter   
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(xgb, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


# XGBoost
xgb = XGBClassifier(n_estimators=50, max_depth=3, min_child_weight=6, gamma=1)

# parameter to be searched
param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1]}

# find the best parameter
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


# XGBoost
xgb = XGBClassifier(n_estimators=50, max_depth=3, min_child_weight=6)

# parameter to be searched
param_grid = {'gamma': [0, 1, 5]}

# find the best parameter   
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(xgb, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


#LightGBM
#LightGBM model, another tree-based learning algorithm, we are going to tune the following hyperparameters:

#max_depth - Maximum depth of a tree
#learning_rate - Shrinkage rate
#num_leaves - Max number of leaves in one tree
#min_data_in_leaf - Minimal number of data in one leaf


# In[ ]:


# LightGBM
lbg = LGBMClassifier(silent=False)

# parameter to be searched
param_grid = {"max_depth": np.arange(5, 75, 10),
              "learning_rate" : [0.001, 0.01, 0.1],
              "num_leaves": np.arange(20, 220, 50),
             }

# find the best parameter            
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(lbg, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


lbg = LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=50, silent=False)

# parameter to be searched
param_grid = {'min_data_in_leaf': np.arange(100, 1000, 100)}

# find the best parameter            
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(lbg, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


#CatBoost
#we're going to search over hyperparameter values for CatBoost, our third gradient boosting algorithm. The following hyperparameters will be tuned:

#depth - Depth of the tree
#learning_rate - As we already know, the learning rate
#l2_leaf_reg - Coefficient at the L2 regularization term of the cost function


# In[58]:


# CatBoost
cb = CatBoostClassifier()

# parameter to be searched
param_grid = {'depth': [6, 8, 10],
              'learning_rate': [0.03, 0.1],
              'l2_leaf_reg': [1, 5, 10],
             }

# find the best parameter            
kfold = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(cb, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# In[ ]:


# final XGBoost model
xgb = XGBClassifier(max_depth=3, learning_rate=0.0001, n_estimators=50, gamma=1, min_child_weight=6)
xgb.fit(X_train_rus, y_train_rus)

# prediction
X_test_xgb = scaler.transform(X_test)
y_pred_xgb = xgb.predict(X_test_xgb)

# classification report
print(classification_report(y_test, y_pred_xgb))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize='true'), annot=True, ax=ax)
ax.set_title('Confusion Matrix - XGBoost')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Real Value')

plt.show()


# In[ ]:


# final LightGBM model
lgb = LGBMClassifier(num_leaves=70, max_depth=5, learning_rate=0.01, min_data_in_leaf=400)
lgb.fit(X_train_rus, y_train_rus)

# prediction
X_test_lgb = scaler.transform(X_test)
y_pred_lgb = lgb.predict(X_test_lgb)

# classification report
print(classification_report(y_test, y_pred_lgb))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_lgb, normalize='true'), annot=True, ax=ax)
ax.set_title('Confusion Matrix - LightGBM')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Real Value')

plt.show()


# In[ ]:


# final CatBoost model
cb = CatBoostClassifier(learning_rate=0.03, depth=6, l2_leaf_reg=5, logging_level='Silent')
cb.fit(X_train_rus, y_train_rus)

# prediction
X_test_cb = scaler.transform(X_test)
y_pred_cb = cb.predict(X_test_cb)

# classification report
print(classification_report(y_test, y_pred_cb))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_cb, normalize='true'), annot=True, ax=ax)
ax.set_title('Confusion Matrix - CatBoost')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Real Value')

plt.show()


# In[44]:


#make a pickle file
pickle.dump(xgb,open("model.pkl","wb"))


# In[45]:


#Conclusion

#The best model possible would be the one that could minimize false negatives, identifying all defaulters among the client base, while also minimizing false positives, 
#preventing clients to be wrongly classified as defaulters.

#Among the three Gradient Boosting Algorithms tested, XGBoost yielded the best results


# In[ ]:




