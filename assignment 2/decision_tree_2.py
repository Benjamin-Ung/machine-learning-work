#-------------------------------------------------------------------------
# AUTHOR: Benjmin Ung
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH 
#AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 
'contact_lens_training_3.csv']
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the
    #4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    Age = {
        "Young": 1,
        "Presbyopic": 2,
        "Prepresbyopic":3
    }
    Spectacle_Prescription = {
        "Myope": 1,
        "Hypermetrope": 2,
    }
    Astigmatism = {
        "No": 1,
        "Yes":2
    }
    Tear_Production = {
        "Normal":1,
        "Reduced":2
    }
    db_Training_copy = [0 for x in range(len(dbTraining))]
    for i in range(len(dbTraining)):
        db_Training_copy[i] = (
            Age[dbTraining[i][0]],
            Spectacle_Prescription[dbTraining[i][1]],
            Astigmatism[dbTraining[i][2]],
            Tear_Production[dbTraining[i][3]]
        )

    X = db_Training_copy
    #transform the original categorical training classes to numbers and add to the 
    #vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    result = {
        "No":1,
        "Yes":2
    }
    dbClass = [0 for x in range(len(dbTraining))]
    for i in range(len(dbTraining)):
        dbClass[i] = result[dbTraining[i][4]]
    Y = dbClass

    counter = 0
    iterations = 10
    #loop your training and test tasks 10 times here
    for i in range (iterations):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)
        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = ['contact_lens_test.csv']
        for data in dbTest:
            dbTestData= []
            with open(data, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i > 0: #skipping the header
                        dbTestData.append (row)

#transform the features of the test instances to numbers following the same strategy done during training,
#and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
#where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        db_Test_copy = [0 for x in range(len(dbTestData))]
        for i in range(len(db_Test_copy)):
            db_Test_copy[i] = (
                Age[dbTestData[i][0]],
                Spectacle_Prescription[dbTestData[i][1]],
                Astigmatism[dbTestData[i][2]],
                Tear_Production[dbTestData[i][3]]
            )

        testX = db_Test_copy

        dbTestClass = [0 for x in range(len(dbTraining))]
        for i in range(len(dbTraining)):
            dbTestClass[i] = result[dbTraining[i][4]]
        testY = dbTestClass
        class_predicted = clf.predict(testX)

#compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        
        for i in range(len(dbTestData)):
            if testY[i] == class_predicted[i]:
                counter += 1
                
#find the average of this model during the 10 runs (training and test set)
    average = float(counter) / float(iterations)
   

#print the average accuracy of this model during the 10 runs (training and test set).
#your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print("The average for this dataset is : " + str(average) )
    #--> add your Python code here
print("done")