from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)


# QUESTION 5
trainedPenData = [] # Make a list to add our finished data to
trainedCarData = [] # Make a list to add our finished data to

for i in range(5): # Iterate five times
    # Test pen data
	neuralNet, error = testPenData()
	trainedPenData.append(error)

    # Test car data
	neuralNet2, error = testCarData()
	trainedCarData.append(error) # Add to carData list

print("===== QUESTION 5 =====")

print ("STATISTICS ON PEN DATA\n", trainedPenData)
print ("Max: ", max(trainedPenData))
print ("Min: ", min(trainedPenData))
print ("Average: ", average(trainedPenData))
print ("Std Dev: ", stDeviation(trainedPenData))

print ("\nSTATISTICS ON CAR DATA\n", trainedCarData)
print ("Max: ", max(trainedCarData))
print ("Min: ", min(trainedCarData))
print ("Average: ", average(trainedCarData))
print ("Std Dev: ", stDeviation(trainedCarData))


# QUESTION 6
pen = {} # Create pen structure
car = {} # Create car structure 
p = 0 # Initialize perceptrons

for p in range(0, 41, 5): # Iterate over 0 to 40 perceptrons, interval of 5
	penData2 = [] # Initialize pen accuracy structure
	carData2 = [] # Initialize car accuracy structure

	for i in range(5): # Iterate five times
        # Test pen data
		neuralNet, error = testPenData(hiddenLayers = [p]) # testPenData with input perceptrons
		penData2.append(error) # Add the error to accuracy list for pen data

        # Test car data
		neuralNet2, error2 = testCarData(hiddenLayers = [p]) # testCarData with input perceptrons
		carData2.append(error2) # Add the error to accuracy list for car data

	penStats = {} # Initialize car stats
	penStats['Max'] = max(penData2) # Calculate max
	penStats['Average'] = average(penData2) # Calculate average
	penStats['Std Dev'] = stDeviation(penData2) # Calculate standard dev
	pen[p] = (penData2, penStats) # Store at that perceptron number in pen

	carStats = {} # Initialize car stats
	carStats['Max'] = max(carData2) # Calculate max
	carStats['Average'] = average(carData2) # Calculate average
	carStats['Std Dev'] = stDeviation(carData2) # Calculate standard dev
	car[p] = (carData2, carStats) # Store at that perceptron number in car


print("===== QUESTION 6 =====") # Header

print ("\nPEN DATA RESULTS ")
for entry in pen : # Iterate over everything in pen structure
	print ("Perceptrons: ", entry) # Print perceptrons count
	print (pen[entry]) # Print each individual entry

print ("\nCAR DATA RESULTS")
for entry in car : # Iterate over everything in car structure
	print ("Perceptrons: ", entry) # Print perceptrons count
	print (car[entry]) # Print each individual entry