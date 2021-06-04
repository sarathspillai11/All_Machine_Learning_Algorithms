from sklearn.preprocessing import LabelEncoder
import bisect
import numpy

def LabelEncode(dataframe):
    print('dataframe length :',dataframe.shape)
    labelencoder = LabelEncoder()
    try:
        colLength = dataframe.shape[1]
    except IndexError:
        colLength = 1
    #print('inside label encoder : ',(dataframe.shape[1]))
    print('dataframe length :', colLength)
    print('######################## inside multiple entity label encoding')
    if(colLength != 1):

        for i in range(colLength):

            #print('before : ',dataframe[:, i])
            dataframe[:, i] = labelencoder.fit_transform(dataframe[:, i].astype(str))
            #print('after : ',dataframe[:,i])
            # dataframe[:, i] = dataframe[:, i].map(lambda s: 'other' if s not in labelencoder.classes_ else s)
            # dataframe[:, i] = numpy.array(list(map(lambda s: 'other' if s not in labelencoder.classes_ else s, dataframe[:, i])))
            # le_classes = labelencoder.classes_.tolist()
            # bisect.insort_left(le_classes, 'other')
            # labelencoder.classes_ = le_classes
            # dataframe[:, i] = labelencoder.transform(dataframe[:, i])
        return dataframe
    else:
        print('######################## inside single entity label encoding')
        dataframe = labelencoder.fit_transform((dataframe))
        # dataframe = numpy.array(list(map(lambda s: 'other' if s not in labelencoder.classes_ else s, dataframe)))
        # le_classes = labelencoder.classes_.tolist()
        # #bisect.insort_left(le_classes, -2)
        # labelencoder.classes_ = le_classes
        # dataframe = labelencoder.transform(dataframe)
        print('y labels : ',labelencoder.classes_)
        return dataframe,labelencoder


