#-------------------------------------------------------------------------
# AUTHOR: Benjamin Ung
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#loop your data to allow each instance to be your test set
error_Xcounter = 0
for x in db:

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    Xcounter = 0
    db_copy = [0 for x in range(len(db) - 1)]
    for i in range(len(db)):
        if db[i] != x:
            db_copy[Xcounter] = (float(db[i][0]), float(db[i][1]) )
            Xcounter += 1
    X = db_copy

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. 
    # Convert each feature value to float to avoid warning messages
    Ycounter = 0
    db_class = [0 for x in range(len(db) - 1)]
    for i in range(len(db)):
        if db[i] != x:
            db_class[Ycounter] = db[i][2]
            Ycounter += 1
    Y = db_class
    #store the test sample of this iteration in the vector testSample
    testSample = [[float(x[0]), float(x[1])]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict(testSample)
    print(class_predicted)

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if(class_predicted != x[2]):
        error_Xcounter += 1

#print the error rate
error_rate = float(error_Xcounter) / (float(len(db)) )
print("The error rate for 1NN: " + str(error_rate))
