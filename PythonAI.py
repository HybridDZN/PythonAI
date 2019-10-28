# Linear Regression Machine Learning, Monday 28 October 2019 2300

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Setting up the data frame, calling the file and defining the separator within the csv.
data = pd.read_csv("student-mat.csv", sep=";")

# Defining the data to be used in the AI
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

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

# Displaying accuracy of the AI calculations
print(round(accuracy*100, 2), "%")

# m coefficient of y=mx+c, shows slope of each different variable used in AI calculations
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # Displays the data the AI is actually calculating
    print(predictions[x], x_test[x], y_test[x])

