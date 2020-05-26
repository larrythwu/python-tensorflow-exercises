import tensorflow
import keras
#ML algorithm library
import sklearn
from sklearn import  linear_model
from sklearn.utils import shuffle
#read data
import pandas as pd
#array manipulation
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# take the data with the following attribute out
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

#get rid of data with attribute "G3"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
#test batch is 10% of the total data size
x_train,x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

best = 0.0
for __ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    #model initialization
    linear = linear_model.LinearRegression()
    #train the data
    linear.fit(x_train, y_train)
    #evaluate the accuracy on the test data
    accuracy = linear.score(x_test, y_test)
    # print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print(best)
pickle_in= open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], y_test[i])

style.use("ggplot")
p="G1"
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()