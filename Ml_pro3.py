# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

#filter the warnings if any
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


####################################################################################################################################################################################################################################################################################################################
# part1 - clustering to obtain labels for the dataset
# Importing the dataset
dataset = pd.read_csv('dataset_small.csv')
X = dataset.iloc[:, :].values
# y = dataset.iloc[:, 3].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in range(4):
    labelencoder = LabelEncoder()
    X[:, i] = labelencoder.fit_transform(X[:, i])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3])
X = onehotencoder.fit_transform(X).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

'''
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
'''

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 13, affinity = 'euclidean', linkage = 'ward')
y = hc.fit_predict(X)


#add the cluster values to the dataset
dataset = pd.read_csv('dataset_small.csv')
Y = dataset.iloc[:, :].values
Y1 = np.hstack((Y, np.atleast_2d(y).T))


clusters = []
for i in range(13):
    clusters.append(list(Y[y == i, :]))
    
from numpy import savez_compressed
savez_compressed('dataset_small_13_clusters.npz', clusters = clusters)


####################################################################################################################################################################################################################################################################################################################
# part2 - classification on the final dataset obtained

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

####################################################################################################################################################################################################################################################################################################################
'''
import pickle
# save the model to disk
pickle.dump(classifier, open('model_1.sav', 'wb'))
pickle.dump(labelencoder, open('label_encoder.sav', 'wb'))
pickle.dump(onehotencoder, open('onehot_encoder.sav', 'wb'))
pickle.dump(sc, open('standard_scalar.sav', 'wb'))

labelencoder = pickle.load(open('label_encoder.sav', 'rb'))
onehotencoder = pickle.load(open('onehot_encoder.sav', 'rb'))
sc = pickle.load(open('standard_scalar.sav', 'rb'))


features = ['fair', 'Black', 'white', 'white']
X = labelencoder.fit_transform(list(features)).reshape(1, -1)
X = onehotencoder.transform(X).toarray()
X = sc.transform(X)


# load the model from disk
loaded_model = pickle.load(open('model_1.sav', 'rb'))
result = loaded_model.predict(X)
print(result)
'''