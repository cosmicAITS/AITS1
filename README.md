# 1. System requirements
These are running environments and may work with lower specs.
## Hardware
OS  : Windows 10<br>
CPU : Intel Core i7-8700K @ 3.70GHz<br>
RAM : 16GB<br>
GPU : NVIDIA Geforce GTX 1660<br>

## Software
The environment was mainly built using anaconda.<br>
conda : 4.9.2<br>
python : 3.8.5<br>
<br>

# 2. Installation guide
Install anaconda from the following URL.<br>
https://www.anaconda.com/products/individual<br>

Run the following command at the command prompt to install the necessary python libraries.
```
$ pip install -r requirements.txt
```
The installation time is about 10 minutes.<br>
<br>

# 3. Instructions for use
The following can be executed in "demo.ipynb".<br>
The run time is about 30 seconds.<br>
<br>

## - Get data
Get training data and testing data.<br>
The data format can be "pandas.dataframe" or "numpy.array".<br>
```python
import pandas as pd
train_data = pd.read_csv(r'.\data\traindata_overtANDlatentThyroidism.csv', encoding='cp932')
test_data = pd.read_csv(r'.\data\testdata_overtThyroidism.csv', encoding='cp932')
```
<br>

## - Extract the required data
Extract the target columns from the data.<br>
Also, extract or exclude specified attributes from the data as needed.<br>
```python
num_features = ['AST', 'ALT', 'γ-GTP', 'Total_cholesterol', 'RBC', 'Hb', 'UA', 'S-Cr', 'UA_S-Cr', 'ALP']
cat_features = ['Sex']
obj_variable = 'class'
info_variable = 'attribute'
target_columns = num_features+cat_features+[obj_variable, info_variable]

#Extract "target_columns"
train_data = train_data.loc[:,target_columns]
test_data = test_data.loc[:,target_columns]

#Exclude "info_variable" including "gunma" from the training data
train_data = train_data[~train_data[info_variable].str.contains('gunma')].reset_index(drop=True)
```
"num_features"  : Numerical features to be input to the machine learning model.<br>
"cat_features"  : Categorical features to be input to the machine learning model.<br>
"obj_variable"  : Teacher labels for machine learning models.<br>
"info_variable" : Attribute information for data extraction.<br>
<br>

## - Apply label encoding
Apply label encoding to categorical features and teacher labels.<br>
```python
#Label encoding("cat_features")
label_encoder = {'male':0, 'female':1}
train_data.loc[:,cat_features] = train_data.loc[:,cat_features].applymap(lambda x: label_encoder[x])
test_data.loc[:,cat_features] = test_data.loc[:,cat_features].applymap(lambda x: label_encoder[x])

#Label encoding("obj_variable")
label_encoder = {'hyper':1, 'hypox':0, 'normal':0}
train_data[obj_variable] = train_data[obj_variable].map(lambda x: label_encoder[x])
test_data[obj_variable] = test_data[obj_variable].map(lambda x: label_encoder[x])
```
<br>

## - Define model
Define model.<br>
Here is an example with "CatBoost".<br>
```python
from catboost import CatBoostClassifier
##Define machine learning model
model = CatBoostClassifier() #Catboost
```
<br>

## - Train and test model
The training data and test data are divided into equal numbers, and training and testing are performed on each divided data.<br>
Here is an example where the data is divided into 10 parts.
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score

kf = StratifiedKFold(n_splits=10)

X_train = train_data.loc[:,num_features+cat_features]
y_train = train_data[obj_variable]
X_test = test_data.loc[:,num_features+cat_features]
y_true = test_data[obj_variable]

result = pd.DataFrame()
for train_indexes, test_indexes in zip(kf.split(X_train, y_train), kf.split(X_test, y_true)):
    train_index = train_indexes[0]
    test_index = test_indexes[1]
    
    model.fit(X_train.loc[train_index,:], y_train[train_index], verbose=0)
    
    proba = model.predict_proba(X_test.loc[test_index,:])
    positive_proba = proba[:,1]
    auroc = roc_auc_score(y_true[test_index], positive_proba)
        
    y_pred = np.where(positive_proba>=0.5, 1, 0)
    cm = confusion_matrix(y_true[test_index], y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.flatten()
        
    recall = tp / (tp+fn)
    specificity = tn / (tn+fp)
    
    result = result.append(pd.Series([auroc, recall, specificity], index=['AUROC', 'Recall', 'Specificity']), ignore_index=True)
```
<br>

# 4. References
- Web sites about software.<br>
https://www.anaconda.com/products/individual<br>
- Web sites about machine learning related python libraries.<br>
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning<br>
https://www.tensorflow.org/api_docs/python/tf/keras?hl=ja<br>


