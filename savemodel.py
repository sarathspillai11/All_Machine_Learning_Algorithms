import pickle
import os



def saveas_sav(model, filename):

    curDirectory = os.getcwd()+os.sep
    filename = curDirectory+filename
    print('saving the model to the current working directory :',filename)
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
    
    
if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    saveas_sav(model,'model.sav')