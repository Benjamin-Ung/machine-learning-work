#-------------------------------------------------------------------------
# AUTHOR: Benjamin Ung
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
dataSet = ["weather_training.csv"]

dbTraining = []
X = []
Y = []
#reading the training data in a csv file
with open(dataSet[0], 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
Outlook = {
     "Sunny":1,
     "Overcast":2,
     "Rain":3
}
Temperature = {
     "Hot":1,
     "Mild":2,
     "Cool":3
}
Humidity = {
     "High":1,
     "Normal":2
}
Wind = {
     "Weak":1,
     "Strong":2
}

db_Training_copy = [0 for x in range(len(dbTraining))]
for i in range(len(dbTraining)):
    db_Training_copy[i] = (
        Outlook[dbTraining[i][1]],
        Temperature[dbTraining[i][2]],
        Humidity[dbTraining[i][3]],
        Wind[dbTraining[i][4]]
    )

X = db_Training_copy

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
trainingClass = {
     "No":1,
     "Yes":2
}
Y = [trainingClass[dbTraining[i][5]] for i in range(len(dbTraining))]

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

testData = ["weather_test.csv"]
dbTest = []
#reading the test data in a csv file
with open(testData[0], 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTest.append (row)

db_Test_copy = [0 for x in range(len(dbTest))]
for i in range(len(dbTest)):
    db_Test_copy[i] = (
        Outlook[dbTraining[i][1]],
        Temperature[dbTraining[i][2]],
        Humidity[dbTraining[i][3]],
        Wind[dbTraining[i][4]]
    )

#printing the header os the solution
print("{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}".format(
     "Day",
     "Outlook",
     "Temperature",
     "Humidity",
     "Wind",
     "Play Tennis",
     "Confidence"))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
result = clf.predict_proba(db_Test_copy)
# print(result)


for i in range(len(result)):
     if(result[i][0] >= 0.75):
          print("{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10f}".format(
               dbTest[i][0], #day
               dbTest[i][1], #outlook
               dbTest[i][2], #temp
               dbTest[i][3], #humidity
               dbTest[i][4], #Wind
               "No",
               result[i][0]
          ))
     elif(result[i][1] >= 0.75):
          print("{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10f}".format(
               dbTest[i][0], #day
               dbTest[i][1], #outlook
               dbTest[i][2], #temp
               dbTest[i][3], #humidity
               dbTest[i][4], #Wind
               "Yes",
               result[i][1]
          ))