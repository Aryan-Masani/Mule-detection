from sklearn import pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv('findataset.csv')

X = df.drop(['ID', 'F3924'], axis=1)
y = df['F3924']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) # Fill holes with Median
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

processor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train = processor.fit_transform(X_train)
X_test = processor.transform(X_test)


#print(f"Before SMOTE: {Counter(y_train)}")

sm = SMOTE(random_state=42, k_neighbors=5)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#print(f"After SMOTE: {Counter(y_train_res)}")

