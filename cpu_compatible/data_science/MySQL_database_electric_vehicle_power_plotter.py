




#Importing own external classes
from classGraphicalAlgorithms import GraphicalAlgorithms
from classDataHandlingAlgorithms import DataHandlingAlgorithms
from classParameters import Parameters
konstanter = Parameters.Constants()
variables = Parameters.Variables()
vectors = Parameters.Vectors()
booleans = Parameters.Booleans()
strings = Parameters.Strings()
GUI = GraphicalAlgorithms.GUI()


#Importing required Python libraries
import numpy as np
import sys
sys.path.insert(1, strings.functionFolder)
import datetime as dt
import os
import shutil
import time

#Launch GUI
var = strings.entryDate;
strings.entryDate = GUI.GUIcreate()
var = strings.entryDate
print("MySQL electric vehicle power plotter");
print("By Olof Brandt Lundqvist")
print("Starting GUI.")

strings.resultsFolder = 'power_log_graphs/' + var + '/';

#Define input variables for data acquisition
userYear = int(var[0]) * 1000 + int(var[1]) * 100 + int(var[2]) * 10 + int(var[3]);
userMonth = int(var[4]) * 10 + int(var[5]);
userDay = int(var[6]) * 10 + int(var[7]);
variables.time1 = dt.datetime(userYear,userMonth,userDay)
variables.time2 = dt.datetime(userYear,userMonth,userDay + 1)


#eTruck power plotter program runs from here
print("Downloading from database.")
#Collecting data from MySQL server
variables, vectors = DataHandlingAlgorithms.getData(variables, konstanter, vectors)

print("Calculating mean power vectors.")
#Function calculating mean power vectors
vectors = DataHandlingAlgorithms.meanPowerCalc(variables, vectors)


#Function detecting when the truck is running on the road in the right direction
print("Identifying test runs on the road.")
vectors = DataHandlingAlgorithms.demobanaPosition(variables, vectors)
vectors.startTimeVec = np.multiply(vectors.startTimeVec, 60 * 60);

#Insert zeros into the position vector, at the beginning of each test run
vectors.fsVec = DataHandlingAlgorithms.fsCalc(vectors.df4, variables.voltstart);


print("Identifying vectors for mean power and traveled distance.")
if (booleans.distancePlot == True):
    vectors = DataHandlingAlgorithms.kilometerCalc(vectors, variables)

#Create a folder for saving the graphs
strings.folderName = 'power_log_graphs/' + var;
if booleans.createFolder==True:
    if os.path.isdir(strings.folderName)==True:
        shutil.rmtree(strings.folderName)
        time.sleep(2);
        os.mkdir(strings.folderName)
    else:
        os.mkdir(strings.folderName)

print("Plotting!")

#Plot all test runs
DataHandlingAlgorithms.plotKilometer(konstanter, variables, vectors, booleans, strings)

