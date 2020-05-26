import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
# preprocessing -> convert test in data file to numerical values

data = pd.read_csv("car.data")
print(data.head())
le = preprocessing.LabelEncoder()
#######Panda -> Numerical List
#fit transform automatically assign a numerical value to the text attributes
#in this case we also fit transformed numerical data such as doors, which essensially just transform array to list
#list: array -> list
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train,x_test,y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Takes in the amount of neighbors
model = KNeighborsClassifier(4);
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", predictions[x], "Actual: ", y_test[x] )
    print("Predicted: ", names[predictions[x]], "Actual: ", names[y_test[x]])
    n=model.kneighbors([x_test[x]], 2, True)
    print(n, "\n")

