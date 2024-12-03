# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score
from keras.utils import np_utils
import numpy as np

import matplotlib.pyplot as plt	 

np.random.seed(321)

#----------------------------------------------------DATA------------------------------------------------------------------------------------------------------

# # load dataset diabetes
# pima = pd.read_csv("diabetes.csv", header=0)

# #split dataset in features and target variable
# feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
# X = pima[feature_cols] # Features
# y = pima.Outcome # Target variable

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# load dataset iris
pima = pd.read_csv("Iris.csv", header=0)

#split dataset in features and target variable
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = pima[feature_cols] # Features
y = pima.Species # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test


#--------------------------------------------Decision Tree-------------------------------------------------------------------------------------------------------
#---------------------------------------- diabetes
# #------------- Learning Curve
# sampleSize = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     # Create Decision Tree classifer object
#     clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
#     clf1 = clf1.fit(X_train2,y_train2)
#     y_pred = clf1.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
#     cv_results = cross_validate(clf2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.plot(sampleSize,trainScoreP)
# plt.plot(sampleSize, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Decision Tree: Learning Curve - Pima Indians Diabetes Database")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/DTSamplePIDD.png")

# #------------- Validation Curve

# maxdepth = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(1, 100):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     maxdepth.append(i)

#     # Create Decision Tree classifer object
#     clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_leaf=5)
#     clf1 = clf1.fit(X_train2,y_train2)
#     y_pred = clf1.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_leaf=5)
#     cv_results = cross_validate(clf2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(maxdepth,trainScoreA)
# plt.plot(maxdepth,valScoreA)
# plt.plot(maxdepth,trainScoreP)
# plt.plot(maxdepth, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Decision Tree: Validation Curve - Pima Indians Diabetes Database")
# plt.xlabel("Maxdepth")
# plt.ylabel("Score")
# plt.savefig("images/DTDepthPIDD.png")

# #------------------Final Test


# clf_Final = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
# cv_resultsFinal = cross_validate(clf_Final, X_train, y_train, cv=5, return_estimator=True)
# gtest_scoreFinal = []
# gtest_scorePFinal = []
# for i in range(len(cv_resultsFinal['estimator'])):
#     gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
#     y_pred_p = cv_resultsFinal['estimator'][i].predict(X_test)
#     gtest_scorePFinal.append(precision_score(y_test, y_pred_p, average='binary'))
# print("DT Accuracy and Precision Diabetes")
# print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))
# print(sum(gtest_scorePFinal) / len(gtest_scorePFinal))


# #------------------------------------------- iris
# #------------- Learning Curve

# sampleSize = []
# trainScoreA = []
# valScoreA = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     # Create Decision Tree classifer object
#     clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
#     clf1 = clf1.fit(X_train2,y_train2)
#     y_pred = clf1.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
#     cv_results = cross_validate(clf2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Decision Tree: Learning Curve - Iris Species")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/DTSampleIS.png")

# #------------- Validation Curve

# maxdepth = []
# trainScoreA = []
# valScoreA = []

# for i in range(1, 100):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     maxdepth.append(i)

#     # Create Decision Tree classifer object
#     clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_leaf=5)
#     clf1 = clf1.fit(X_train2,y_train2)
#     y_pred = clf1.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_leaf=5)
#     cv_results = cross_validate(clf2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(maxdepth,trainScoreA)
# plt.plot(maxdepth,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Decision Tree: Validation Curve - Iris Species")
# plt.xlabel("Maxdepth")
# plt.ylabel("Score")
# plt.savefig("images/DTDepthIS.png")

# #------------------Final Test

clf_Final = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
cv_resultsFinal = cross_validate(clf_Final, X_train, y_train, cv=5, return_estimator=True)
gtest_scoreFinal = []
for i in range(len(cv_resultsFinal['estimator'])):
    gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
print("DT Accuracy Iris")
print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))




#----------------------------------------------------end--------------------------------------------------------------------------------------

#----------------------------------------------------Neural networks--------------------------------------------------------------------------
#---------------------------------------- diabetes
# #------------- Learning Curve

# epoch = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(50, 200, 10):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 70% training and 30% test
#     epoch.append(i)

#     model = Sequential()
#     model.add(Dense(15, input_shape=(8,), activation='relu'))
#     model.add(Dense(12, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # compile the keras model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train2, y_train2, epochs=i, batch_size=10, verbose=0)


#     y_pred = (model.predict(X_train2) > 0.5).astype(int)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))

#     y_pred = (model.predict(X_test2) > 0.5).astype(int)
#     valScoreA.append(metrics.accuracy_score(y_test2, y_pred))
#     valScoreP.append(precision_score(y_test2, y_pred, average='binary'))

# plt.clf()
# plt.plot(epoch,trainScoreA)
# plt.plot(epoch,valScoreA)
# plt.plot(epoch,trainScoreP)
# plt.plot(epoch, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Neural Networks: Learning Curve - Pima Indians Diabetes Database")
# plt.xlabel("Epoch")
# plt.ylabel("Score")
# plt.savefig("images/NNEpochPIDD.png")

# #------------- Validation Curve

# layers = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(5, 50, 5):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     layers.append(i)

#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), activation='relu'))
#     model.add(Dense(i, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # compile the keras model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train2, y_train2, epochs=170, batch_size=10, verbose=0)


#     y_pred = (model.predict(X_train2) > 0.5).astype(int)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))

#     y_pred = (model.predict(X_test2) > 0.5).astype(int)
#     valScoreA.append(metrics.accuracy_score(y_test2, y_pred))
#     valScoreP.append(precision_score(y_test2, y_pred, average='binary'))

# plt.clf()
# plt.plot(layers,trainScoreA)
# plt.plot(layers,valScoreA)
# plt.plot(layers,trainScoreP)
# plt.plot(layers, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Neural Networks: Validation Curve - Pima Indians Diabetes Database")
# plt.xlabel("Number of Node in Middle layer")
# plt.ylabel("Score")
# plt.savefig("images/NNLayersPIDD.png")

# #------------------Final Test


# model = Sequential()
# model.add(Dense(12, input_shape=(8,), activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=170, batch_size=10, verbose=0)

# y_pred = (model.predict(X_test) > 0.5).astype(int)
# print("Neural Networks Accuracy and Precision Diabetes")
# print(metrics.accuracy_score(y_test, y_pred))
# print(precision_score(y_test, y_pred, average='binary'))

# ------------------------------------------- iris
#---------------- Learning Curve

# epoch = []
# trainScoreA = []
# valScoreA = []

# for i in range(10, 50, 10):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 70% training and 30% test
#     epoch.append(i)

#     model=Sequential()
#     model.add(Dense(1000,input_dim=4,activation='relu'))
#     model.add(Dense(500,activation='relu'))
#     model.add(Dense(300,activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(3,activation='softmax'))
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     # fit the keras model on the dataset

#     y_train_new = []
#     for j in y_train2:
#         if j == "Iris-setosa":
#             y_train_new.append(0)
#         elif j == "Iris-versicolor":
#             y_train_new.append(1)
#         elif j == "Iris-virginica":
#             y_train_new.append(2)

#     y_test_new = []
#     for j in y_test2:
#         if j == "Iris-setosa":
#             y_test_new.append(0)
#         elif j == "Iris-versicolor":
#             y_test_new.append(1)
#         elif j == "Iris-virginica":
#             y_test_new.append(2)

#     y_train2=np_utils.to_categorical(y_train_new,num_classes=3)
#     y_test2=np_utils.to_categorical(y_test_new,num_classes=3)

#     model.fit(X_train2,y_train2,validation_data=(X_test2,y_test2),batch_size=20,epochs=i,verbose=0)


#     y_pred = (model.predict(X_train2) > 0.5).astype(int)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))

#     y_pred = (model.predict(X_test2) > 0.5).astype(int)
#     valScoreA.append(metrics.accuracy_score(y_test2, y_pred))

# plt.clf()
# plt.plot(epoch,trainScoreA)
# plt.plot(epoch,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Neural Networks: Learning Curve - Iris Species")
# plt.xlabel("Epoch")
# plt.ylabel("Score")
# plt.savefig("images/NNEpochIS.png")

# #------------- Validation Curve

# layers = []
# trainScoreA = []
# valScoreA = []

# for i in range(100, 900, 50):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     layers.append(i)

#     model=Sequential()
#     model.add(Dense(1000,input_dim=4,activation='relu'))
#     model.add(Dense(i,activation='relu'))
#     model.add(Dense(300,activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(3,activation='softmax'))
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     # fit the keras model on the dataset

#     y_train_new = []
#     for j in y_train2:
#         if j == "Iris-setosa":
#             y_train_new.append(0)
#         elif j == "Iris-versicolor":
#             y_train_new.append(1)
#         elif j == "Iris-virginica":
#             y_train_new.append(2)

#     y_test_new = []
#     for j in y_test2:
#         if j == "Iris-setosa":
#             y_test_new.append(0)
#         elif j == "Iris-versicolor":
#             y_test_new.append(1)
#         elif j == "Iris-virginica":
#             y_test_new.append(2)

#     y_train2=np_utils.to_categorical(y_train_new,num_classes=3)
#     y_test2=np_utils.to_categorical(y_test_new,num_classes=3)

#     model.fit(X_train2,y_train2,validation_data=(X_test2,y_test2),batch_size=20,epochs=20,verbose=0)


#     y_pred = (model.predict(X_train2) > 0.5).astype(int)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))

#     y_pred = (model.predict(X_test2) > 0.5).astype(int)
#     valScoreA.append(metrics.accuracy_score(y_test2, y_pred))

# plt.clf()
# plt.plot(layers,trainScoreA)
# plt.plot(layers,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Neural Networks: Validation Curve - Iris Species")
# plt.xlabel("Number of Node in Middle layer")
# plt.ylabel("Score")
# plt.savefig("images/NNLayersIS.png")

# #------------------Final Test


# model=Sequential()
# model.add(Dense(1000,input_dim=4,activation='relu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(300,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# # fit the keras model on the dataset

# y_train_new = []
# for i in y_train:
#     if i == "Iris-setosa":
#         y_train_new.append(0)
#     elif i == "Iris-versicolor":
#         y_train_new.append(1)
#     elif i == "Iris-virginica":
#         y_train_new.append(2)

# y_test_new = []
# for i in y_test:
#     if i == "Iris-setosa":
#         y_test_new.append(0)
#     elif i == "Iris-versicolor":
#         y_test_new.append(1)
#     elif i == "Iris-virginica":
#         y_test_new.append(2)

# y_train=np_utils.to_categorical(y_train_new,num_classes=3)
# y_test=np_utils.to_categorical(y_test_new,num_classes=3)

# model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=20,verbose=0)

# y_pred = (model.predict(X_test) > 0.5).astype(int)
# print("Neural Networks Accuracy Iris")
# print(metrics.accuracy_score(y_test, y_pred))

#----------------------------------------------------end--------------------------------------------------------------------------------------

#----------------------------------------------------K-nearest Neighbors--------------------------------------------------------------------------
#---------------------------------------- diabetes
# #------------- Learning Curve
# sampleSize = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     knn = KNeighborsClassifier(n_neighbors=7)
#     knn = knn.fit(X_train2,y_train2)
#     y_pred = knn.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     knn2 = KNeighborsClassifier(n_neighbors=7)
#     cv_results = cross_validate(knn2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.plot(sampleSize,trainScoreP)
# plt.plot(sampleSize, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("KNN: Learning Curve - Pima Indians Diabetes Database")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/KNNSamplePIDD.png")

# #------------- Validation Curve

# neighbors = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(1, 100):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     neighbors.append(i)

#     # Create Decision Tree classifer object
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn = knn.fit(X_train2,y_train2)
#     y_pred = knn.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     knn2 = KNeighborsClassifier(n_neighbors=i)
#     cv_results = cross_validate(knn2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(neighbors,trainScoreA)
# plt.plot(neighbors,valScoreA)
# plt.plot(neighbors,trainScoreP)
# plt.plot(neighbors, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("KNN: Validation Curve - Pima Indians Diabetes Database")
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Score")
# plt.savefig("images/KNNNeighborsPIDD.png")

# #------------------Final Test


# KNN_Final = KNeighborsClassifier(n_neighbors=7)
# cv_resultsFinal = cross_validate(KNN_Final, X_train, y_train, cv=5, return_estimator=True)
# gtest_scoreFinal = []
# gtest_scorePFinal = []
# for i in range(len(cv_resultsFinal['estimator'])):
#     gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
#     y_pred_p = cv_resultsFinal['estimator'][i].predict(X_test)
#     gtest_scorePFinal.append(precision_score(y_test, y_pred_p, average='binary'))
# print("KNN Accuracy and Precision Diabetes")
# print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))
# print(sum(gtest_scorePFinal) / len(gtest_scorePFinal))

# # ------------------------------------------- iris
# #------------- Learning Curve

# sampleSize = []
# trainScoreA = []
# valScoreA = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     knn = KNeighborsClassifier(n_neighbors=7)
#     knn = knn.fit(X_train2,y_train2)
#     y_pred = knn.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     knn2 = KNeighborsClassifier(n_neighbors=7)
#     cv_results = cross_validate(knn2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("KNN: Learning Curve - Iris Species")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/KNNSampleIS.png")

# #------------- Validation Curve

# neighbors = []
# trainScoreA = []
# valScoreA = []

# for i in range(1, 50):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     neighbors.append(i)

#     # Create Decision Tree classifer object
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn = knn.fit(X_train2,y_train2)
#     y_pred = knn.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     knn2 = KNeighborsClassifier(n_neighbors=i)
#     cv_results = cross_validate(knn2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(neighbors,trainScoreA)
# plt.plot(neighbors,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("KNN: Validation Curve - Iris Species")
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Score")
# plt.savefig("images/KNNNeighborsIS.png")

# #------------------Final Test

knn_Final = KNeighborsClassifier(n_neighbors=7)
cv_resultsFinal = cross_validate(knn_Final, X_train, y_train, cv=5, return_estimator=True)
gtest_scoreFinal = []
for i in range(len(cv_resultsFinal['estimator'])):
    gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
print("KNN Accuracy Iris")
print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))

#----------------------------------------------------end--------------------------------------------------------------------------------------

#----------------------------------------------------SVM------------------------------------------------------------------------------------------------
#---------------------------------------- diabetes
#------------- Learning Curve
# sampleSize = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     svmModel = svm.SVC(kernel='linear')
#     svmModel = svmModel.fit(X_train2,y_train2)
#     y_pred = svmModel.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     svmModel2 = svm.SVC(kernel='linear')
#     cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.plot(sampleSize,trainScoreP)
# plt.plot(sampleSize, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("SVM: Learning Curve - Pima Indians Diabetes Database")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/SVMSamplePIDD.png")

# #------------- Validation Curve

# Score = []
# Names = ["Training Accuracy - Linear", "Training Precision - Linear", "Validation Accuracy - Linear", "Validation Precision - Linear",
#          "Training Accuracy - Poly", "Training Precision - Poly", "Validation Accuracy - Poly", "Validation Precision - Poly",
#          "Training Accuracy - rbf", "Training Precision - rbf", "Validation Accuracy - rbf", "Validation Precision - rbf",
#          "Training Accuracy - Sigmoid", "Training Precision - Sigmoid", "Validation Accuracy - Sigmoid", "Validation Precision - Sigmoid"]

# # Split dataset into training set and test set
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test

# #Linear
# svmModel = svm.SVC(kernel='linear')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))
# Score.append(precision_score(y_train2, y_pred, average='binary'))


# svmModel2 = svm.SVC(kernel='linear')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# gtest_scoreP = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     y_pred_p = cv_results['estimator'][i].predict(X_test2)
#     gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
# Score.append(sum(gtest_score) / len(gtest_score))
# Score.append(sum(gtest_scoreP) / len(gtest_scoreP))

# #Poly
# svmModel = svm.SVC(kernel='poly')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))
# Score.append(precision_score(y_train2, y_pred, average='binary'))


# svmModel2 = svm.SVC(kernel='poly')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# gtest_scoreP = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     y_pred_p = cv_results['estimator'][i].predict(X_test2)
#     gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
# Score.append(sum(gtest_score) / len(gtest_score))
# Score.append(sum(gtest_scoreP) / len(gtest_scoreP))

# #rbf
# svmModel = svm.SVC(kernel='rbf')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))
# Score.append(precision_score(y_train2, y_pred, average='binary'))


# svmModel2 = svm.SVC(kernel='rbf')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# gtest_scoreP = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     y_pred_p = cv_results['estimator'][i].predict(X_test2)
#     gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
# Score.append(sum(gtest_score) / len(gtest_score))
# Score.append(sum(gtest_scoreP) / len(gtest_scoreP))

# #Sigmoid
# svmModel = svm.SVC(kernel='sigmoid')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))
# Score.append(precision_score(y_train2, y_pred, average='binary'))


# svmModel2 = svm.SVC(kernel='sigmoid')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# gtest_scoreP = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     y_pred_p = cv_results['estimator'][i].predict(X_test2)
#     gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
# Score.append(sum(gtest_score) / len(gtest_score))
# Score.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.figure(figsize=(20, 10))
# plt.xticks(rotation=45, ha='right')
# plt.bar(Names, Score)
# plt.title("SVM: Validation Curve - Pima Indians Diabetes Database")
# plt.xlabel("Kernel Used")
# plt.ylabel("Score")
# plt.savefig("images/SVMKernelPIDD.png", bbox_inches="tight")

# #------------------Final Test


# svm_Final = svm.SVC(kernel='linear')
# cv_resultsFinal = cross_validate(svm_Final, X_train, y_train, cv=5, return_estimator=True)
# gtest_scoreFinal = []
# gtest_scorePFinal = []
# for i in range(len(cv_resultsFinal['estimator'])):
#     gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
#     y_pred_p = cv_resultsFinal['estimator'][i].predict(X_test)
#     gtest_scorePFinal.append(precision_score(y_test, y_pred_p, average='binary'))
# print("SVM Accuracy and Precision Diabetes")
# print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))
# print(sum(gtest_scorePFinal) / len(gtest_scorePFinal))

# #------------------------------------------- iris
# #------------- Learning Curve

# sampleSize = []
# trainScoreA = []
# valScoreA = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     svmModel = svm.SVC(kernel='linear')
#     svmModel = svmModel.fit(X_train2,y_train2)
#     y_pred = svmModel.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     svmModel2 = svm.SVC(kernel='linear')
#     cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("SVM: Learning Curve - Iris Species")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/SVMSampleIS.png")

# #------------- Validation Curve

# Score = []
# Names = ["Training Accuracy - Linear", "Validation Accuracy - Linear",
#          "Training Accuracy - Poly", "Validation Accuracy - Poly",
#          "Training Accuracy - rbf", "Validation Accuracy - rbf",
#          "Training Accuracy - Sigmoid", "Validation Accuracy - Sigmoid"]

# # Split dataset into training set and test set
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test

# #Linear
# svmModel = svm.SVC(kernel='linear')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))


# svmModel2 = svm.SVC(kernel='linear')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
# Score.append(sum(gtest_score) / len(gtest_score))

# #Poly
# svmModel = svm.SVC(kernel='poly')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))


# svmModel2 = svm.SVC(kernel='poly')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
# Score.append(sum(gtest_score) / len(gtest_score))

# #rbf
# svmModel = svm.SVC(kernel='rbf')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))


# svmModel2 = svm.SVC(kernel='rbf')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
# Score.append(sum(gtest_score) / len(gtest_score))

# #Sigmoid
# svmModel = svm.SVC(kernel='sigmoid')
# svmModel = svmModel.fit(X_train2,y_train2)
# y_pred = svmModel.predict(X_train2)
# Score.append(metrics.accuracy_score(y_train2, y_pred))


# svmModel2 = svm.SVC(kernel='sigmoid')
# cv_results = cross_validate(svmModel2, X_train2, y_train2, cv=5, return_estimator=True)
# gtest_score = []
# for i in range(len(cv_results['estimator'])):
#     gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
# Score.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.figure(figsize=(20, 10))
# plt.xticks(rotation=45, ha='right')
# plt.bar(Names, Score)
# plt.title("SVM: Validation Curve - Iris Species")
# plt.xlabel("Kernel Used")
# plt.ylabel("Score")
# plt.savefig("images/SVMKernelIS.png", bbox_inches="tight")

# #------------------Final Test


svm_Final = svm.SVC(kernel='linear')
cv_resultsFinal = cross_validate(svm_Final, X_train, y_train, cv=5, return_estimator=True)
gtest_scoreFinal = []
for i in range(len(cv_resultsFinal['estimator'])):
    gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
print("SVM Accuracy Iris")
print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))


#----------------------------------------------------end--------------------------------------------------------------------------------------

#----------------------------------------------------Boosting--------------------------------------------------------------------------------------
#---------------------------------------- diabetes
# #------------- Learning Curve
# sampleSize = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     abc = AdaBoostClassifier(n_estimators=150,learning_rate=1)
#     abc = abc.fit(X_train2,y_train2)
#     y_pred = abc.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     abc2 = AdaBoostClassifier(n_estimators=150,learning_rate=1)
#     cv_results = cross_validate(abc2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.plot(sampleSize,trainScoreP)
# plt.plot(sampleSize, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Boosting: Learning Curve - Pima Indians Diabetes Database")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/BoostingSamplePIDD.png")

# #------------- Validation Curve

# weaklearner = []
# trainScoreA = []
# valScoreA = []
# trainScoreP = []
# valScoreP = []

# for i in range(1, 300, 5):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     weaklearner.append(i)

#     # Create Decision Tree classifer object
#     abc = AdaBoostClassifier(n_estimators=i,learning_rate=1)
#     abc = abc.fit(X_train2,y_train2)
#     y_pred = abc.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))
#     trainScoreP.append(precision_score(y_train2, y_pred, average='binary'))


#     abc2 = AdaBoostClassifier(n_estimators=i,learning_rate=1)
#     cv_results = cross_validate(abc2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     gtest_scoreP = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#         y_pred_p = cv_results['estimator'][i].predict(X_test2)
#         gtest_scoreP.append(precision_score(y_test2, y_pred_p, average='binary'))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))
#     valScoreP.append(sum(gtest_scoreP) / len(gtest_scoreP))

# plt.clf()
# plt.plot(weaklearner,trainScoreA)
# plt.plot(weaklearner,valScoreA)
# plt.plot(weaklearner,trainScoreP)
# plt.plot(weaklearner, valScoreP)
# plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
# plt.title("Boosting: Validation Curve - Pima Indians Diabetes Database")
# plt.xlabel("Number of Weak Learners")
# plt.ylabel("Score")
# plt.savefig("images/BoostingWeaklearnerPIDD.png")

# #------------------Final Test


# abc_Final = AdaBoostClassifier(n_estimators=150,learning_rate=1)
# cv_resultsFinal = cross_validate(abc_Final, X_train, y_train, cv=5, return_estimator=True)
# gtest_scoreFinal = []
# gtest_scorePFinal = []
# for i in range(len(cv_resultsFinal['estimator'])):
#     gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
#     y_pred_p = cv_resultsFinal['estimator'][i].predict(X_test)
#     gtest_scorePFinal.append(precision_score(y_test, y_pred_p, average='binary'))
# print("Boosting Accuracy and Precision Diabetes")
# print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))
# print(sum(gtest_scorePFinal) / len(gtest_scorePFinal))


# # ------------------------------------------- iris
# #------------- Learning Curve

# sampleSize = []
# trainScoreA = []
# valScoreA = []

# for i in range(9, 0, -1):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=i/10, random_state=1) # 70% training and 30% test
#     sampleSize.append((10-i)*10)

#     abc = AdaBoostClassifier(n_estimators=150,learning_rate=1)
#     abc = abc.fit(X_train2,y_train2)
#     y_pred = abc.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     abc2 = AdaBoostClassifier(n_estimators=150,learning_rate=1)
#     cv_results = cross_validate(abc2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(sampleSize,trainScoreA)
# plt.plot(sampleSize,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Boosting: Learning Curve - Iris Species")
# plt.xlabel("Sample Size")
# plt.ylabel("Score")
# plt.savefig("images/BoostingSampleIS.png")

# #------------- Validation Curve

# weaklearner = []
# trainScoreA = []
# valScoreA = []

# for i in range(1, 300, 5):
#     # Split dataset into training set and test set
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=1) # 70% training and 30% test
#     weaklearner.append(i)

#     # Create Decision Tree classifer object
#     abc = AdaBoostClassifier(n_estimators=i,learning_rate=1)
#     abc = abc.fit(X_train2,y_train2)
#     y_pred = abc.predict(X_train2)
#     trainScoreA.append(metrics.accuracy_score(y_train2, y_pred))


#     abc2 = AdaBoostClassifier(n_estimators=i,learning_rate=1)
#     cv_results = cross_validate(abc2, X_train2, y_train2, cv=5, return_estimator=True)
#     gtest_score = []
#     for i in range(len(cv_results['estimator'])):
#         gtest_score.append(cv_results['estimator'][i].score(X_test2, y_test2))
#     valScoreA.append(sum(gtest_score) / len(gtest_score))

# plt.clf()
# plt.plot(weaklearner,trainScoreA)
# plt.plot(weaklearner,valScoreA)
# plt.legend(["Training - Accuracy", "Validation - Accuracy"])
# plt.title("Boosting: Validation Curve - Iris Species")
# plt.xlabel("Number of Weaklearners")
# plt.ylabel("Score")
# plt.savefig("images/BoostingWeaklearnerIS.png")

#------------------Final Test

abc_Final = AdaBoostClassifier(n_estimators=150,learning_rate=1)
cv_resultsFinal = cross_validate(abc_Final, X_train, y_train, cv=5, return_estimator=True)
gtest_scoreFinal = []
for i in range(len(cv_resultsFinal['estimator'])):
    gtest_scoreFinal.append(cv_resultsFinal['estimator'][i].score(X_test, y_test))
print("Boosting Accuracy Iris")
print(sum(gtest_scoreFinal) / len(gtest_scoreFinal))


#----------------------------------------------------end--------------------------------------------------------------------------------------

"""
-------------------------------------------Sources----------------------------------------------
Dataset
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
https://www.kaggle.com/datasets/uciml/iris

Code
https://www.datacamp.com/tutorial/decision-tree-classification-python
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://www.kaggle.com/code/louisong97/neural-network-approach-to-iris-dataset
https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://www.datacamp.com/tutorial/adaboost-classifier-python
https://inside-machinelearning.com/en/cross-validation-tutorial/
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.precision_score.html
https://saturncloud.io/blog/how-does-scikitlearn-compute-the-precision-score-metric/
"""