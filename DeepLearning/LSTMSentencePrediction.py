import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys, re, os, shutil
from savemodel import saveas_sav

def LSTMPredictor(inputTextFile,ticketId):


    # load ascii text and covert to lowercase
    #filename = r"D:\Personal\SmartIT\data\wonderland.txt"
    filename = inputTextFile
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # The problem is really a single character classification problem with 47 classes and as such is defined as optimizing the log loss (cross entropy), here using the ADAM optimization algorithm for speed.

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # load the network weights

    curDirectory = os.getcwd()
    try:
        os.mkdir('NNWeights')
    except:
        print('already exists.. hence cleaning up')
        print('all files :', list(os.walk(curDirectory + os.sep + 'NNWeights')))

        for root, dirs, files in os.walk(curDirectory + os.sep + 'NNWeights'):
            for file in files:
                os.remove(os.path.join(root, file))

    # define the checkpoint
    filepath = curDirectory + os.sep + "NNWeights" + os.sep + r"weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit our model to the data. Here we use a modest number of 20 epochs and a large batch size of 128 patterns.
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

    weightFiles = (list(os.walk(curDirectory + os.sep + 'NNWeights'))[0])[-1]
    temp = weightFiles
    weightFiles = list(enumerate(weightFiles))
    weightFiles = [(file[23:29], index) for (index, file) in weightFiles]
    weightFiles.sort(key=lambda x: x[0])
    weightFileIndex = (weightFiles[0])[1]
    lowLossWeightFile = temp[weightFileIndex]

    # Generating Text with an LSTM Network

    # load the network weights
    filename = curDirectory + os.sep + 'NNWeights' + os.sep + lowLossWeightFile
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # Making predictions

    resultList = []
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    print('start :', start)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        # print('actual prediction :',prediction)
        index = numpy.argmax(prediction)
        # print('argmax : ',index)
        result = int_to_char[index]
        # print('converted to char :',result)
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        resultList.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

    predictedText = ''.join(resultList)
    saveas_sav(model, 'LSTM_' + ticketId + '.sav')
    return predictedText