def transformData(dataframe,encoderObject):
    y_pred = list(dataframe['predicted'])
    for i in range((dataframe.shape[1])-1):
        dataframe[:, i] = encoderObject.inverse_transform(dataframe[:, i])

    return dataframe