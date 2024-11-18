import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

data=pd.read_csv('training_data.csv', sep=',')
test_data=pd.read_csv('songs_to_classify.csv', sep=',')
k = 10

# select which features to use
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

test_data = test_data.iloc[:].values

# use chatgpt to delete outlier
Iso = IsolationForest(contamination=0.02)     
outliers = Iso.fit_predict(X) 
outlier_indices = np.where(outliers == -1)
X = np.delete(X, outlier_indices, axis=0)
y = np.delete(y, outlier_indices, axis=0)

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test_data = scaler.fit_transform(test_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)

# evaluate
score = model.score(X_test, y_test)
print("accuracy", score)

# K-fold accuracy
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print('K-fold accuracy:', scores)

# predict
scaler = MinMaxScaler()
normalize_test_data = scaler.fit_transform(test_data)
predicted_labels = model.predict(normalize_test_data)
# print("Predicted Labels:", predicted_labels)

print(str(predicted_labels.reshape(1,200).tolist()).replace(',', '').replace(' ', ''))