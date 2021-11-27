from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame= False)
# print(mnist.keys())
X, y = mnist['data'], mnist['target']
# print(X.shape)
# print(y.shape)
#############################################

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
y = y.astype(np.uint8)
# print(y[0])

# plt.imshow(some_digit_image, cmap="binary")
# plt.axis("off")
# plt.show()
X_train, X_test, y_tain, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_tain==5)
y_test_5 = (y_test==5)

#################################################

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#################################################

# from not_5class import Never5Classifier


# nev_5_clf = Never5Classifier()
# print(cross_val_score(nev_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#################################################

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_train_5, y_train_pred))

y_train_perfect_predictions = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_predictions))

from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))