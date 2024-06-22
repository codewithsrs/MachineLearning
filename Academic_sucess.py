import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
train_data = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")
original = pd.read_csv("/kaggle/input/original/data.csv",delimiter = ';')
train_data.head()

original = original.rename({'Daytime/evening attendance\t':'Daytime/evening attendance'}, axis = 1)
original.head()

original.shape

train_data.describe()

train_data = train_data.drop('id', axis = 1)
train_data = pd.concat([train_data, original])
test = test.drop('id', axis = 1)
train_data.head()

numeric_cols = ['GDP','Inflation rate','Unemployment rate',\
               'Curricular units 2nd sem (grade)','Curricular units 2nd sem (approved)',\
               'Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (enrolled)',\
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',\
                'Curricular units 2nd sem (without evaluations)',
               'Curricular units 1st sem (enrolled)',\
                'Curricular units 1st sem (evaluations)',\
                'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)','Age at enrollment',\
               'Admission grade','Previous qualification (grade)']

dropped = []
for i in numeric_cols:
    if  np.percentile(train_data[i],25)==0 \
    and np.percentile(train_data[i],50)==0 and np.percentile(train_data[i],75)==0:
        train_data = train_data.drop(i, axis = 1)
        dropped.append(i)
        print(i)
    else:
        continue

for i in sorted(numeric_cols):
    if i in sorted(dropped):
        numeric_cols.remove(i)

cat_cols = [i for i  in train_data.columns if i not in  numeric_cols and train_data[i].nunique()!= 2]
others = [i for i in train_data.columns if train_data[i].nunique()==2]

print(len(train_data.columns))
for i in train_data.columns:

    if i not in numeric_cols and i not in cat_cols and i not in others:
        train_data = train_data.drop(i, axis = 1)
print(len(train_data.columns))

for i in test.columns:
    if i not in train_data.columns:
        test = test.drop(i, axis = 1)

train_data.drop('Target', axis = 1).columns==test.columns

def outlier_treat(df, col):
    q1 = np.percentile( df[col],25)
    q3 = np.percentile(df[col],75)
    print(f"{col} q1 is {q1} and q3 is {q3}")

    iqr = q3 - q1
    print(f"{col} inter quartile range is {iqr}")
    low = q1-(iqr*1.5)
    print(f"{col} lower bound for outlier is {low}")
    up = q3+(iqr*1.5)
    print(f"{col} upper bound for outlier is {up}")
    df[col] = np.where(df[col]<=low,low,np.where(df[col]>=up,up,df[col]))
#     plt.boxplot(df[col])
#     plt.show()

for col in numeric_cols:
    outlier_treat(train_data,col)
    outlier_treat(test, col)
    print('-'*100)

for i in cat_cols:
    if i == 'Target':
        continue 
    elif train_data[i].nunique()!=test[i].nunique():
        print(i)

test.describe()

from scipy import stats
import seaborn as sns
# for col in col_to_remove_outlier:
#     print(stats.skew(train_data[col]))
#     print(stats.skew(stats.boxcox(0.00001+train_data[col])[0]))
#     print("raw")
#     sns.kdeplot(train_data[col])
#     plt.show()
    
for col in numeric_cols:
    if abs(stats.skew(train_data[col])) >= 0.5:
        print(f"{col} original skewness : {+stats.skew(train_data[col])}")
        print(f"{col} boxcox skewness : {stats.skew(stats.boxcox(0.00000000001+train_data[col])[0])}")
        train_data[col] = stats.boxcox(0.00000000001+train_data[col])[0]
    else:
        continue
        
print('-'*100)

for col in numeric_cols:
    if abs(stats.skew(test[col])) >= 0.5:
        print(f"{col} original skewness : {+stats.skew(test[col])}")
        print(f"{col} boxcox skewness : {stats.skew(stats.boxcox(0.00000000001+test[col])[0])}")
        test[col] = stats.boxcox(0.00000000001+test[col])[0]
    else:
        continue

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
for i in numeric_cols:
    train_data[i] = s.fit_transform(train_data[[i]])
    test[i] = s.transform(test[[i]])

for i in sorted(cat_cols):
    if i!='Target' and len(train_data[i].unique()) < len(test[i].unique()):
        print(len(train_data[i].unique()))
        print(len(test[i].unique()))
        print(train_data.shape,test.shape)
        train_data = train_data.drop(i, axis = 1)
        test = test.drop(i, axis = 1)
        print(train_data.shape,test.shape)
        print(i)
    else:
        continue

for i in sorted(cat_cols):
    if i not in train_data.columns:
        cat_cols.remove(i)
        print(len(cat_cols))

for col in cat_cols:
    if col == 'Target':
        continue
    else:
        total_feats = set(train_data[col].unique())|set(test[col].unique())
        common_feats = set(train_data[col].unique())&set(test[col].unique())
        missing_in_both = total_feats-common_feats
        train_data[col] = np.where(train_data[col].isin(missing_in_both),0,train_data[col])
        test[col] = np.where(test[col].isin(missing_in_both),0,test[col])

train_data= train_data.drop('Nacionality', axis = 1)
test= test.drop('Nacionality', axis = 1)

cat_cols.remove('Nacionality')

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
cat_cols.remove('Target')
for column in cat_cols:
    train_data[column] = lb.fit_transform(train_data[column])
    train_data[column] = train_data[column].astype('category')
      
    test[column] = lb.transform(test[column])
    test[column] = test[column].astype('category')

train_data.head()

x = train_data.drop(['Target'], axis = 1)
y = lb.fit_transform(train_data['Target'])

for col in others:
    x[col] = x[col].astype('bool')
    test[col] = test[col].astype('bool')

x.head()

import xgboost as xgb

xg = xgb.XGBClassifier(enable_categorical = True, random_state = 3 )

from sklearn.model_selection import train_test_split
train_x, test_x, train_y,  test_y = train_test_split(x, y, test_size = 0.2)



xg = xg.fit(train_x,train_y)

prec = xg.predict(test_x)

from sklearn.metrics import f1_score, accuracy_score
f1_score(test_y, prec,average = 'weighted')
print(accuracy_score(test_y,prec))

pred = xg.predict(test)
sample = pd.read_csv("/kaggle/input/playground-series-s4e6/sample_submission.csv")
pred = lb.inverse_transform(pred)
sample['Target'] = pred

sample

# %% [code] {"execution":{"iopub.status.busy":"2024-06-22T04:36:23.502479Z","iopub.execute_input":"2024-06-22T04:36:23.503167Z","iopub.status.idle":"2024-06-22T04:36:23.581541Z","shell.execute_reply.started":"2024-06-22T04:36:23.503129Z","shell.execute_reply":"2024-06-22T04:36:23.580781Z"},"jupyter":{"outputs_hidden":false}}
sample.to_csv("submission.csv", index = False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
