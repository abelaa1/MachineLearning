from sklearn.model_selection import train_test_split
import sklearn
from sklearn import mixture
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt	 

pima = pd.read_csv("Iris.csv", header=0)

#split dataset in features and target variable
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = pima[feature_cols] # Features
y = pima.Species # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test

# load dataset diabetes
pima = pd.read_csv("diabetes.csv", header=0)

#split dataset in features and target variable
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = pima[feature_cols] # Features
y = pima.Outcome # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


np.random.seed(321)
#---------------------------------------------------------Clustering-EM--------------------------------------------------------------------------------
#-----------------------EM Iris-------------------------------
name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": 2
}


gm = mixture.GaussianMixture(n_components=3)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)
print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_fixed==0, 0:3]
x1 = x_new[y_fixed==1, 0:3]
x2 = x_new[y_fixed==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("Correct Results - Iris")
plt.savefig("images3/CorrectIR.png")


x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("Expectation Maximization - Iris")
plt.savefig("images3/EM_IR.png")


#---------------------------EM PIDD---------------------------

gm = mixture.GaussianMixture(n_components=2)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[np.array(y_train)==0, 0:7]
x1 = x_new[np.array(y_train)==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("Correct Results - PIDD")
plt.savefig("images3/CorrectPIDD.png")


x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("Expectation Maximization - PIDD")
plt.savefig("images3/EM_PIDD.png")

#-----------------------------------------------------------Clustering Kmeans------------------------------------------------------------------------------
#---------------------------Kmeans Iris---------------------------
name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": 2
}

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)
print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("Kmeans - Iris")
plt.savefig("images3/Kmeans_IR.png")

#---------------------------Kmeans PIDD---------------------------

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("Kmeans - PIDD")
plt.savefig("images3/Kmeans_PIDD.png")

#----------------------------------------------------------PCA-------------------------------------------------------------------------------
name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": 2
}

y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)

pca = PCA(n_components = 4)

X_train = pca.fit_transform(X_train)

x_new = np.array(X_train)

x0 = x_new[y_fixed==0, 0:3]
x1 = x_new[y_fixed==1, 0:3]
x2 = x_new[y_fixed==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("PCA - Correct Results - Iris")
plt.savefig("images3/PCACorrectIR.png")

#-----------------------Iris-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=3)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("PCA EM - Iris")
plt.savefig("images3/PCA_EM_IR.png")


#--------------------Iris-Kmeans---------------------------------- 

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("PCA Kmeans - Iris")
plt.savefig("images3/PCA_Kmeans_IR.png")


#------------------------------------------------------------ 
pca = PCA(n_components = 8)
X_train = pca.fit_transform(X_train)

x_new = np.array(X_train)

x0 = x_new[np.array(y_train)==0, 0:7]
x1 = x_new[np.array(y_train)==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("PCA - Correct Results - PIDD")
plt.savefig("images3/PCACorrectPIDD.png")

#----------------------PIDD-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=2)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("PCA - Expectation Maximization - PIDD")
plt.savefig("images3/PCA_EM_PIDD.png")

#-----------------------PIDD-Kmeans------------------------------- 

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("PCA - Kmeans - PIDD")
plt.savefig("images3/PCA_Kmeans_PIDD.png")

#--------------------------------------------------------ICA---------------------------------------------------------------------------------

ica = FastICA(n_components=4)
X_train = ica.fit_transform(X_train)

name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": 2
}
y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)

x_new = np.array(X_train)

x0 = x_new[y_fixed==0, 0:3]
x1 = x_new[y_fixed==1, 0:3]
x2 = x_new[y_fixed==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ICA - Correct Results - Iris")
plt.savefig("images3/ICACorrectIR.png")
#-----------------------Iris-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=3)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ICA EM - Iris")
plt.savefig("images3/ICA_EM_IR.png")

#--------------------Iris-Kmeans---------------------------------- 

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ICA Kmeans - Iris")
plt.savefig("images3/ICA_Kmeans_IR.png")

#------------------------------------------------------

ica = FastICA(n_components=8)
X_train = ica.fit_transform(X_train)

x_new = np.array(X_train)

x0 = x_new[np.array(y_train)==0, 0:7]
x1 = x_new[np.array(y_train)==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ICA - Correct Results - PIDD")
plt.savefig("images3/ICACorrectPIDD.png")

#----------------------PIDD-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=2)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ICA - Expectation Maximization - PIDD")
plt.savefig("images3/ICA_EM_PIDD.png")

#-----------------------PIDD-Kmeans------------------------------- 

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ICA - Kmeans - PIDD")
plt.savefig("images3/ICA_Kmeans_PIDD.png")
#--------------------------------------------------------------Random-Projection---------------------------------------------------------------------------

grp = GaussianRandomProjection(n_components=4)
X_train = grp.fit_transform(X_train)

name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 0
}

y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)

x_new = np.array(X_train)

x0 = x_new[y_fixed==0, 0:3]
x1 = x_new[y_fixed==1, 0:3]
x2 = x_new[y_fixed==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("RP - Correct Results - Iris")
plt.savefig("images3/RPCorrectIR.png")
#-----------------------Iris-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=3)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("RP EM - Iris")
# plt.savefig("images3/RP_EM_IR.png")---------------------------------------

# -------------------------------------------------------Knapsack--------------------------------------------------------------------------------------------
#--------------------Iris-Kmeans---------------------------------- 

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("RP Kmeans - Iris")
plt.savefig("images3/RP_Kmeans_IR.png")

#------------------------------------------------------

grp = GaussianRandomProjection(n_components=8)
X_train = grp.fit_transform(X_train)

x_new = np.array(X_train)

x0 = x_new[np.array(y_train)==0, 0:7]
x1 = x_new[np.array(y_train)==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("RP - Correct Results - PIDD")
plt.savefig("images3/RPCorrectPIDD.png")

#----------------------PIDD-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=2)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("RP - Expectation Maximization - PIDD")
plt.savefig("images3/RP_EM_PIDD.png")

#-----------------------PIDD-Kmeans------------------------------- 

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("RP - Kmeans - PIDD")
plt.savefig("images3/RP_Kmeans_PIDD.png")
#-------------------------------------------------------------ISOMAP----------------------------------------------------------------------------

isomap = Isomap(n_components=4) 
X_train = isomap.fit_transform(X_train)

name_to_num = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": 2
}
y_fixed = [name_to_num[name] for name in y_train]
y_fixed = np.array(y_fixed)

x_new = np.array(X_train)

x0 = x_new[y_fixed==0, 0:3]
x1 = x_new[y_fixed==1, 0:3]
x2 = x_new[y_fixed==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ISOMAP - Correct Results - Iris")
plt.savefig("images3/ISOMAPCorrectIR.png")

#-----------------------Iris-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=3)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ISOMAP EM - Iris")
plt.savefig("images3/ISOMAP_EM_IR.png")

#--------------------Iris-Kmeans---------------------------------- 

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

print(metrics.accuracy_score(y_fixed, y_pred))

x_new = np.array(X_train)

x0 = x_new[y_pred==0, 0:3]
x1 = x_new[y_pred==1, 0:3]
x2 = x_new[y_pred==2, 0:3]

plt.clf()
plt.scatter(x0[:,0], x0[:,1], c='navy')
plt.scatter(x1[:,0], x1[:,1], c='g')
plt.scatter(x2[:,0], x2[:,1], c='orange')
plt.legend(["Iris-versicolor", "Iris-setosa", "Iris-virginica"])
plt.title("ISOMAP Kmeans - Iris")
plt.savefig("images3/ISOMAP_Kmeans_IR.png")

#------------------------------------------------------

isomap = Isomap(n_components=8) 
X_train = isomap.fit_transform(X_train)

x_new = np.array(X_train)

x0 = x_new[np.array(y_train)==0, 0:7]
x1 = x_new[np.array(y_train)==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ISOMAP - Correct Results - PIDD")
plt.savefig("images3/ISOMAPCorrectPIDD.png")

#----------------------PIDD-EM------------------------------- 

gm = mixture.GaussianMixture(n_components=2)

gm.fit(X_train)

y_pred = gm.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ISOMAP - Expectation Maximization - PIDD")
plt.savefig("images3/ISOMAP_EM_PIDD.png")

#-----------------------PIDD-Kmeans-------------------------------

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
y_pred = np.array(y_pred)

x_new = np.array(X_train)
print(metrics.accuracy_score(y_train, y_pred))

x0 = x_new[y_pred==0, 0:7]
x1 = x_new[y_pred==1, 0:7]

plt.clf()
plt.scatter(x0[:,0], x0[:,2], c='navy')
plt.scatter(x1[:,0], x1[:,2], c='g')
plt.legend(["Negative", "Positive"])
plt.title("ISOMAP - Kmeans - PIDD")
plt.savefig("images3/ISOMAP_Kmeans_PIDD.png")

#-------------------------------------------------------NeuralNetworks----------------------------------------------------------------------------------
#-----------------------PCA-------------------------------

pca = PCA(n_components = 8)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(8,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(X_train) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: PCA Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_PCA_PIDD.png")

#-------------------------ICA-----------------------------

ica = FastICA(n_components=8)

X_train = ica.fit_transform(X_train)
X_test = ica.transform(X_test)

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(8,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(X_train) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: ICA Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_ICA_PIDD.png")

#----------------------------RP--------------------------

grp = GaussianRandomProjection(n_components=8)

X_train = grp.fit_transform(X_train)
X_test = grp.transform(X_test)

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(8,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(X_train) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: RP Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_RP_PIDD.png")

#------------------------------ISOMAP------------------------

isomap = Isomap(n_components=8) 

X_train = isomap.fit_transform(X_train)
X_test = isomap.transform(X_test)

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(8,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(X_train) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: ISOMAP Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_ISOMAP_PIDD.png")

#-----------------------------------------------------NeuralNetworks-Part2-----------------------------------------------------------------------------------
#------------------------------EM------------------------

gm = mixture.GaussianMixture(n_components=2)
gm.fit(X_train)

y_pred = gm.predict(X_train)
x_nn = np.insert(X_train, 0, y_pred, axis=1)

y_pred_test = gm.predict(X_test)
x_nn_test = np.insert(X_test, 0, y_pred_test, axis=1)

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(9,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_nn, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(x_nn) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(x_nn_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: EM Extra Feature Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_EMF_PIDD.png")

#------------------------------Kmeans------------------------

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_train)
x_nn = np.insert(X_train, 0, y_pred, axis=1)

y_pred_test = kmeans.predict(X_test)
x_nn_test = np.insert(X_test, 0, y_pred_test, axis=1)


epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 200, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(9,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_nn, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(x_nn) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(x_nn_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: Kmeans Extra Feature Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images3/NN_KmeansF_PIDD.png")

#----------------------------------------------Sources-------------------------------------------------------------------------
"""
-------------------------------------------Sources----------------------------------------------
Dataset
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
https://www.kaggle.com/datasets/uciml/iris

Code
https://scikit-learn.org/stable/modules/manifold.html
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection
https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
https://scikit-learn.org/stable/modules/mixture.html
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://www.w3schools.com/python/python_ml_k-means.asp
https://www.geeksforgeeks.org/principal-component-analysis-with-python/
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://www.geeksforgeeks.org/blind-source-separation-using-fastica-in-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.random_projection.gaussianrandomprojection
https://www.geeksforgeeks.org/comparison-of-manifold-learning-methods-in-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
"""