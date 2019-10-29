# Linear Regression Machine Learning, Monday 28 October 2019 2300

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Setting up the data frame, calling the file and defining the separator within the csv.
data = pd.read_csv("student-mat.csv", sep=";")

# Defining the data to be used in the AI
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# Shuffles data (optional)
data = shuffle(data)

# Known as a label, this is what we are trying to 'get'.
predict = "G3"

# Array to define attribute
X = np.array(data.drop([predict], 1))
# Array to define label
y = np.array(data[predict])

# Taking the attributes and labels, splitting them up into four different arrays
# Test data will test the accuracy of the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Training model
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print("AI Output:\n ")
# m coefficient of y=mx+c, shows slope of each different variable used in AI calculations
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # Displays the data the AI is actually calculating
    print("Computer Result: ", predictions[x], "Test Paramaters: ", x_test[x], "Test Result: ", y_test[x])


# Displaying accuracy of the AI calculations
print("\n AI Prediction: ", round(accuracy * 100, 2), "%")

# Linear is the name of the model that has been created
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentgrades.pickle", "rb")
linear =pickle.load(pickle_in)

# Training the model multiple times for the optimal result
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print("Accuracy: " + str(accuracy))

    # If the current model has a better score than the one that has already been trained then save it
    if accuracy > best:
        best = accuracy
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# Drawing and plotting the model
plot = "failures" # Change this to the previous paramaters defined in data to see the other graphs
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()


