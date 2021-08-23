# import packages
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
from timeit import default_timer as timer
import psutil
import resource
import memory_profiler
from memory_profiler import profile

def uso(mensagem):
    #usage = resource.getrusage(resource.RUSAGE_SELF)
    #print ("**********************************************")
    print (mensagem)
    print ("CPU Usage: " + str(psutil.cpu_times()))
    print ("CPU Percent:None: " + str(psutil.cpu_percent(interval=None, percpu=True)))
    #print ("CPU Percent:1: " + str(psutil.cpu_percent(interval=1, percpu=True)))
    #print ("CPU Percent:0.1: " + str(psutil.cpu_percent(interval=0.1, percpu=True)))
    mem_usage = memory_profiler.memory_usage()[0]
    print ("**** Memory Usage: " + str(mem_usage))
    #for name, desc in [
    # ('ru_utime', 'User time'),
    # ('ru_stime', 'System time'),
    # ('ru_maxrss', 'Max. Resident Set Size'),
    # ('ru_ixrss', 'Shared Memory Size'),
    # ('ru_idrss', 'Unshared Memory Size'),
    # ('ru_isrss', 'Stack Size'),
    # ('ru_inblock', 'Block inputs'),
    # ('ru_oublock', 'Block outputs'),
    # ]:
    # print '%-25s (%-10s) = %s' % (desc, name, getattr(usage, name))

# filter warning messages
import warnings
warnings.filterwarnings('ignore')

# import data set and create a data frame
uso ("import data set and create a data frame: BEGIN")
df_credit = pd.read_csv('http://dl.dropboxusercontent.com/s/xn2a4kzf0zer0xu/acquisition_train.csv?dl=0')
uso ("import data set and create a data frame: END")

uso ("Credit  DataSet size")
print (df_credit.shape[0])

# show first 5 rows
uso ("Head: show first 5 rows")
df_credit.head()

# data frame shape
uso ("data frame shape")
print('Number of rows: ', df_credit.shape[0])
print('Number of columns: ', df_credit.shape[1])

# data frame summary
uso ("data frame summary")
df_credit.info()

# percentage of missing values per feature
uso("percentage of missing values per feature")
print((df_credit.isnull().sum() * 100 / df_credit.shape[0]).sort_values(ascending=False))

df_credit.dropna(subset=['target_default'], inplace=True)


# drop the column "target_fraud"
df_credit.drop('target_fraud', axis=1, inplace=True)

# number of unique observations per column
df_credit.nunique().sort_values()

# drop the columns "channel" and "external_data_provider_credit_checks_last_2_year"
df_credit.drop(labels=['channel', 'external_data_provider_credit_checks_last_2_year'], axis=1, inplace=True)

df_credit.drop(labels=['email', 'reason', 'zip', 'job_name', 'external_data_provider_first_name', 'lat_lon',
                       'shipping_zip_code', 'user_agent', 'profile_tags', 'marketing_channel',
                       'profile_phone_number', 'application_time_applied', 'ids'], axis=1, inplace=True)
                       
# show descriptive statistics
uso ("show descriptive statistics")
df_credit.describe()


uso ("count of inf values in reported_income")
np.isinf(df_credit['reported_income']).sum()

uso ("count of values = -999 in external_data_provider_email_seen_before")
df_credit.loc[df_credit['external_data_provider_email_seen_before'] == -999, 'external_data_provider_email_seen_before'].value_counts()

uso ("replace inf values with nan")
df_credit['reported_income'] = df_credit['reported_income'].replace(np.inf, np.nan)

uso ("replace -999 values with nan")
df_credit.loc[df_credit['external_data_provider_email_seen_before'] == -999, 'external_data_provider_email_seen_before'] = np.nan

uso("data frame containing numerical features") 
df_credit_numerical = df_credit[['score_3', 'risk_rate', 'last_amount_borrowed', 'last_borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_accounts', 'n_issues', 'external_data_provider_email_seen_before']]
                                 
uso("Excluding objects") 
df_credit_num = df_credit.select_dtypes(exclude=object).columns
#                                         \/
df_credit_cat = df_credit.select_dtypes(exclude=object).columns

uso ("fill missing values for last_amount_borrowed, last_borrowed_in_months and n_issues")
df_credit['last_amount_borrowed'].fillna(value=0, inplace=True)
df_credit['last_borrowed_in_months'].fillna(value=0, inplace=True)
df_credit['n_issues'].fillna(value=0, inplace=True)

uso("fill missing values for numerical variables")
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(df_credit.loc[:, df_credit_num])
df_credit.loc[:, df_credit_num] = imputer.transform(df_credit.loc[:, df_credit_num])

uso("fill missing values for categorical variables")
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(df_credit.loc[:, df_credit_cat])
df_credit.loc[:, df_credit_cat] = imputer.transform(df_credit.loc[:, df_credit_cat])


df_credit.isnull().sum()


bin_var = df_credit.nunique()[df_credit.nunique() == 2].keys().tolist()
num_var = [col for col in df_credit.select_dtypes(['int', 'float']).columns.tolist() if col not in bin_var]
cat_var = [col for col in df_credit.select_dtypes(['object']).columns.tolist() if col not in bin_var]

df_credit_encoded = df_credit.copy()

uso ("label encoding for the binary variables")
le = LabelEncoder()
for col in bin_var:
  df_credit_encoded[col] = le.fit_transform(df_credit_encoded[col])

uso("encoding with get_dummies for the categorical variables")
df_credit_encoded = pd.get_dummies(df_credit_encoded, columns=cat_var)

uso("head encoded")
df_credit_encoded.head()


#******************** split the data into training and test sets

uso ("split the data into training and test sets: BEGIN")
# feature matrix
X = df_credit_encoded.drop('target_default', axis=1)
uso ("split the data into training and test sets: END")

uso("target vector") 
y = df_credit_encoded['target_default']

uso("train: BEGIN") 
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)
uso("train: END") 

uso("standardize numerical variables");
scaler = StandardScaler().fit(X_train)

uso("Transformer");
X_train = scaler.transform(X_train)

uso ("resample: BEGIN")
rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_sample(X_train, y_train)
uso ("resample: END")

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
    uso ("make X array")
    X = np.array(X)
    uso ("make y array")
    y = np.array(y)

    uso ("make pipeline")
    pipeline = make_pipeline(StandardScaler(), clf)
    uso ("Calcula Cross Val")
    scores = cross_val_score(pipeline, X, y, scoring='recall')

    if show == True:
        print("Recall:")
        print (scores.mean())
        print (scores.std())
    
    return scores.mean()
    
    
    
#evaluate the models
uso ("evaluate the models")
uso ("XGBClassifier: BEGIN")
xgb = XGBClassifier()
uso ("XGBClassifier: END")

model = []
recall = []

#for clf in (xgb, lgb, cb):
#for clf in (xgb):
#    model.append(clf.__class__.__name__)
#    recall.append(val_model(X_train_rus, y_train_rus, clf, show=False))
#
#pd.DataFrame(data=recall, index=model, columns=['Recall'])


# XGBoost
uso ("XGBoost: BEGIN")
xgb = XGBClassifier()
uso ("XGBoost: END")

# parameter to be searched
uso ("parameter to be searched")
param_grid = {'n_estimators': range(0,1000,50)}

# find the best parameter  
uso ("find the best parameter: kfold")
kfold = StratifiedKFold(n_splits=3, shuffle=True)
uso ("find the best parameter: grid_search")
grid_search = GridSearchCV(xgb, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
uso ("find the best parameter: grid_result")
grid_result = grid_search.fit(X_train_rus, y_train_rus)

uso("Best result")
print(grid_result.best_score_)
print(grid_result.best_params_)



begin_classifier = timer()

uso ("final XGBoost model: BEGIN")
xgb = XGBClassifier(max_depth=3, learning_rate=0.0001, n_estimators=50, gamma=1, min_child_weight=6)
uso ("final XGBoost model: END")

uso ("fit: BEGIN")
xgb.fit(X_train_rus, y_train_rus)
uso ("fit: END")

end_classifier = timer()

# prediction

begin_prediction = timer()

uso ("prediction test: BEGIN")
X_test_xgb = scaler.transform(X_test)
uso ("prediction test: END")

uso ("prediction pred: BEGIN")
y_pred_xgb = xgb.predict(X_test_xgb)
uso ("prediction pred: END")

# classification report
end_prediction = timer()

uso ("classification report")
print(classification_report(y_test, y_pred_xgb))

uso ("Classifier Time in ms");
total_classifier = end_classifier - begin_classifier
print (total_classifier*1000)

print ('Prediction Time in ms');
total_prediction = end_prediction - begin_prediction
print (total_prediction*1000)
# confusion matrix
#print ('confusion matrix')
#fig, ax = plt.subplots()
#sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize='true'), annot=True, ax=ax)
#ax.set_title('Confusion Matrix - XGBoost')
#ax.set_xlabel('Predicted Value')
#ax.set_ylabel('Real Value')

#plt.show()
