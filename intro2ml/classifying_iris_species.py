from sklearn.datasets import load_iris
iris_dataset = load_iris()
import numpy as np
# print("Keys of iris_dataset: \n {}".format(iris_dataset.keys()))

# print(iris_dataset['DESCR'][:193] + "\n...")

# print("Target names: {}".format(iris_dataset['target_names']))


print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

# print(X_train)
# print(y_train)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


X_new = np.array([[5, 2.9, 1, 0.2]])

print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))

print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))