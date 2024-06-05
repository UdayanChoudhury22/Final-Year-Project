import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('D:\water_potability.csv')
# data.describe()
# data.info()


data = data.fillna(data.mean())
# count = data.isnull().sum()
# print(data.head())
# print(data['Turbidity'].min()," ",data['Turbidity'].max())
# print(data.columns)


# dimentionality reduction:Checking the interdependence of features using heat map to reduce them
# plt.figure(figsize=(10, 6))
# sns.heatmap(data.corr(),annot=True,cmap = 'terrain')
# plt.show()

# outlier removal...
# data.boxplot(figsize=(10,6))
# plt.show()
# print(data.describe())
# checking the min max and average values of the solid coloumn
# print(data['Solids'].describe())


# checking the balanace of the output column
# plt.figure(figsize=(10,6))
# sns.countplot(x = 'Potability',data = data,palette='viridis')
# plt.xlabel('Potability')  # Set the x-axis label
# plt.ylabel('Count')  # Set the y-axis label
# plt.xticks(ticks=[0, 1], labels=['Non-Potable', 'Potable'])  # Set x-axis ticks and labels
# plt.show()

# normality of data
# data.hist(figsize=(16,10))
# plt.show()

# Partitioning of the data into training and testing....
x = data.drop('Potability',axis = 1)
y = data['Potability']
# print(x)
# print(y)
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,shuffle=True,random_state=None)
# model training...

# Decision tree
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(x_train,y_train)
# decision_tree = dt.predict(x_test)
# decision_tree_score = accuracy_score(decision_tree,y_test)*100
# print(decision_tree_score)
# cm = confusion_matrix(decision_tree,y_test)






# Random forest....
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
random_forest = rf.predict(x_test)
random_forest_score = accuracy_score(random_forest,y_test)*100
print(f'Random Forest Accuracy: {random_forest_score:.2f}%')


joblib.dump(rf, 'model.pkl')

# knn algorithm
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# knn.fit(x_train,y_train)
# K_neighbors = knn.predict(x_test)
# KNN_score = accuracy_score(K_neighbors,y_test)*100
# print(KNN_score)


# svm algorithm..
# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(x_train,y_train)
# support_vector = svm.predict(x_test)
# svm_score = accuracy_score(support_vector,y_test)*100
# print(svm_score)

# record = {
#     # 'Decision_tree':decision_tree_score,
#     # 'liner_regression':linear_regression_score,
#     'random_forest':random_forest_score
#     # 'KNN':KNN_score,
#     # 'SVM':svm_score
# }

# print(record)

# plt.figure(figsize=(10,6))
# sns.heatmap(cm,annot=True,cmap='Blues',cbar=False,fmt='d')
# plt.show()

# Hyperparameter tuning....

# 0998