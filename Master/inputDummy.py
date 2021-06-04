import pandas as pd

def checkInputs(inputDict):

    inputValues = inputDict.values()
    inputKeys = inputDict.keys()
    allStrings = [inputDict[item] for item in inputKeys if (item != 'dataframe')]
    if(any([item == None for item in allStrings])):
        print('some attribute is none')
        return False

    if(not(isinstance(inputDict['dataframe'],pd.DataFrame))):
        print('not dataframe')
        return False
    elif(any([not(isinstance(item,str)) for item in allStrings])):
        print('not string')
        return False
    return True



if __name__ == '__main__':
    inputDict = {'ticketId': '', 'dataframe':  pd.DataFrame(columns=['CombinationID','StepNo','StepDescription','Info'])
                , 'replaceMissing': 'True', 'dropCardinal': 'True', 'ifPCA': 'True',
                'pcaComp': '3', 'X_train': 'null', 'X_test': 'null',
                'y_train': 'null', 'y_test': 'null', 'inputType': 'labelled', 'contentType': 'text',
                'mlType': 'supervised', 'encodingType': 'one_hot', 'scalingType': 'standard',
                'usecaseType': 'classification', 'custom': '', 'numClusters': '2', 'epsilon': '0.3', 'minSamples': '10'}

    valid = checkInputs(inputDict)

    print('validity : ',valid)