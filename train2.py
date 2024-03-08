import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

# Data Pre Processing
data = pd.read_csv(r'E:\Codes\DOS_PR\New folder\CVD\CVD_cleaned.csv')
print(data.head())
data = data.dropna()

print('Total rows = ',len(data.index))
Health = {'Very Poor':1,'Poor':2, 'Fair':3, 'Good':4 , 'Very Good':5}
Bool = {'Yes':1 , 'No':0}
Sex = {'Male':1 , 'Female':0}
Time = {'Within the past 2 years': 1 , 'Within the past year' : 2}
data['Exercise'] = data['Exercise'].map(Bool)
data['Heart_Disease'] = data['Heart_Disease'].map(Bool)
data['Skin_Cancer'] = data['Skin_Cancer'].map(Bool)
data['Depression'] = data['Depression'].map(Bool)
data['Diabetes'] = data['Diabetes'].map(Bool)
data['Arthritis'] = data['Arthritis'].map(Bool)
data['Other_Cancer'] = data['Other_Cancer'].map(Bool)
data['Smoking_History'] = data['Smoking_History'].map(Bool)
data['Sex'] = data['Sex'].map(Sex)
data['General_Health'] = data['General_Health'].map(Health)
data['Checkup'] = data['Checkup'].map(Time)
data['Age_Category'] = data['Age_Category'].str[:2].astype(int)

data.dropna()
data.round()

# Data Analysis
print(data.describe())
print(data.info())
plt.figure(figsize=(17,6))
sns.heatmap(data.corr(), annot=True)

# Data Cleaning
# data.to_csv('CVD1.csv', index=False)

data = pd.read_csv(r'E:\Codes\DOS_PR\New folder\CVD\CVD1.csv')
data.isnull()
data.dropna()
data.isnull().sum().sum()
data2 = data.fillna(value=0)
data2.isnull().sum()

# Train Test Split
y = data2['Heart_Disease']
X = data2.drop('Heart_Disease', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Feature Engineering

#Feature Selection using Anova
sel = SelectKBest(f_classif, k=10).fit(X_train, y_train)
print(X_train.columns[sel.get_support()])
columns = X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Feature Selection using PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Model Building and Evaluation
# Upsampling using Smote
sm = SMOTE(random_state = 2)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

# Model = Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
lr_predict = model.predict(X_train)
train_accuracy = accuracy_score(lr_predict,y_train)
print(train_accuracy)

# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating a DNN model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1:])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=3, batch_size=64, validation_split=0.1)
test_loss = model.evaluate(X_test_scaled, y_test)
predictions = model.predict(X_test_scaled)