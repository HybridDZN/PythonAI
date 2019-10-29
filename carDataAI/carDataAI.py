from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head()) # Checking if data is loaded correctly

# Converting data to int types
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# Recombine data into a feature list and a label list
# Features
x = list(zip(buying, maint, door, persons, lug_boot, safety))
# Labels
y = list(cls)

# Split data into training and test data using the same process as in First AI.py
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
# Creating the model and defining the final result for accuracy
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)


predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# This will display the predicted class, the AI data and the actual class
# We will create a names list so that we can convert out integer predictions into their string of representation
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
