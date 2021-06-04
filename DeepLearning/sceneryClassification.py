import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec


def cnnPreprocessing(trainFolder):
    Images,Labels = [],[]
    label = 0
    sceneLabels = list(os.listdir(trainFolder))
    numlabels = range(len(sceneLabels))
    sceneLabelsDict = dict(zip(sceneLabels,numlabels))
    
    for labels in sceneLabels:
        for image_file in os.listdir(trainFolder+os.sep+labels):
            image = cv2.imread(trainFolder+os.sep+labels+os.sep+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(sceneLabelsDict[labels])
            
    shuffledDataset = shuffle(Images,Labels,random_state=817328462)
    return shuffledDataset,sceneLabelsDict
    
def cnnPredPreprocessing(predFolder):
    Images,Labels = [],[]
    label = 0
    sceneLabels = list(os.listdir(predFolder))
    numlabels = range(len(sceneLabels))
    sceneLabelsDict = dict(zip(sceneLabels,numlabels))
    
    for labels in sceneLabels:
        for image_file in os.listdir(predFolder):
            image = cv2.imread(predFolder+os.sep+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(sceneLabelsDict[labels])
            
    shuffledDataset = shuffle(Images,Labels,random_state=817328462)
    return shuffledDataset,sceneLabelsDict
def predictionImageProcessor(predFolder):
    Images = []
    for image_file in os.listdir(predFolder):
        image = cv2.imread(predFolder+os.sep+image_file) #Reading the image (OpenCV)
        image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
        Images.append(image)
    shuffledImages = shuffle(Images,random_state=817328462)
    return shuffledImages
    
def SceneClassifier(shuffledTrainset,sceneLabelsDict,shuffledTestSet,shuffledPredImages):
    Images,Labels = shuffledTrainset
    Images = np.array(Images) #converting the list of images to numpy array.
    Labels = np.array(Labels)
    print("Shape of Images:",Images.shape)
    print("Shape of Labels:",Labels.shape)
    
    model = Models.Sequential()

    model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(5,5))
    model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(5,5))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(180,activation='relu'))
    model.add(Layers.Dense(100,activation='relu'))
    model.add(Layers.Dense(50,activation='relu'))
    model.add(Layers.Dropout(rate=0.5))
    model.add(Layers.Dense(6,activation='softmax'))

    model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    model.summary()
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #Utils.plot_model(model,to_file='model.png',show_shapes=True)
    
    trained = model.fit(Images,Labels,epochs=2,validation_split=0.30)
    
    print('history dict :',trained.history)
    
    plot.plot(trained.history['accuracy'])
    plot.plot(trained.history['val_accuracy'])
    plot.title('Model accuracy')
    plot.ylabel('Accuracy')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Test'], loc='upper left')
    plot.show()
    
    plot.plot(trained.history['loss'])
    plot.plot(trained.history['val_loss'])
    plot.title('Model loss')
    plot.ylabel('Loss')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Test'], loc='upper left')
    plot.show()
    
    print('Running the model on test set to check accuracy ......')
    
    test_images,test_labels = shuffledTestSet
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    model.evaluate(test_images,test_labels, verbose=1)
    pred_images = shuffledPredImages
    pred_images = np.array(pred_images)
    
    fig = plot.figure(figsize=(30, 30))
    outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)
    
    for i in range(25):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        rnd_number = randint(0,len(pred_images))
        pred_image = np.array([pred_images[rnd_number]])
        pred_class = sceneLabelsDict[model.predict_classes(pred_image)[0]]
        pred_prob = model.predict(pred_image).reshape(6)
        for j in range(2):
            if (j%2) == 0:
                ax = plot.Subplot(fig, inner[j])
                ax.imshow(pred_image[0])
                ax.set_title(pred_class)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
            else:
                ax = plot.Subplot(fig, inner[j])
                ax.bar([0,1,2,3,4,5],pred_prob)
                fig.add_subplot(ax)
    
    
    fig.show()
    
shuffledTrainset,sceneLabelsDict =  cnnPreprocessing(r'D:\Personal\SmartIT\data\seg_train\seg_train')
shuffledTestSet,temp = cnnPreprocessing(r'D:\Personal\SmartIT\data\seg_test\seg_test')
shuffledPredImages,no_labels = cnnPredPreprocessing(r'D:\Personal\SmartIT\data\seg_pred\seg_pred')

print(shuffledPredImages)

SceneClassifier(shuffledTrainset,sceneLabelsDict,shuffledTestSet,shuffledPredImages)


    
    
    
    