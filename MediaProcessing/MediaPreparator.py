import cv2
import moviepy.editor as mp
import speech_recognition as sr
import pytesseract
import os
import shutil
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\KPMG\AppData\Local\Tesseract-OCR\tesseract.exe"

def video2audio(sourcePath):
    #sourcePath = r'dnc-2004-speech.mp4'  # Path where the videos are located
    clip = mp.VideoFileClip(sourcePath)
    clip.audio.write_audiofile("theaudio.wav")
    r = sr.Recognizer()

    hellow = sr.AudioFile('theaudio.wav')
    with hellow as source:
        audio = r.record(source)
    print('audio extracted ! ')
    return audio

def audioToText(audio):
    try:
        r = sr.Recognizer()
        audiosText = r.recognize_google(audio)
        print("Text: " + audiosText)
        file = open('data.txt', 'w')
        file.write(audiosText)
        file.close()
        return audiosText
    except Exception as e:
        print("Exception: " + str(e))


def videoToFrames(filePath,sourceFlag):
    images = []
    vidcap = cv2.VideoCapture(filePath)
    success, image = vidcap.read()

    images.append(image)
    print(success)
    count = 0
    if(os.path.exists('sourceImages') and sourceFlag == True):
        print('source already exists.. so removing')
        os.rename('sourceImages','sourceImages_old')
    if (os.path.exists('lookupImages') and sourceFlag == False):
        os.rename('lookupImages','lookupImages_old')
    if(sourceFlag==True):
        os.mkdir('sourceImages')
    else:
        os.mkdir('lookupImages')

    while success:
        print('going')

        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()

        if(sourceFlag==True):
            shutil.move(os.getcwd()+os.sep+"frame%d.jpg" % count,'sourceImages')
        else:
            shutil.move(os.getcwd()+os.sep+"frame%d.jpg" % count, 'lookupImages')
        images.append(image)
        print('Read a new frame: ', success)
        count += 1
    return images

def imageToText(inImageList):
    print('in for extracting text fro images')
    texts = []
    count = 0
    for image in inImageList:
        count+=1
        print('file name :',image)
        try:
            text = pytesseract.image_to_string(Image.open(image),lang='eng')
        except Exception as e:
            print(e)
            text = ''
        print('text found :',text)
        print(count,len(inImageList))
        texts.append(text)

    print('all texts :',texts)
    return texts

def sentimentAnalysis(text):
    pass

def mediaPreparation(sourcePath,sourceType,lookupPath,lookupType):
    """

    :param sourcePath: exact location of source file including filename
    :param sourceType: can be either audio, video, image or text
    :param lookupPath: exact location of destination file including filename
    :param lookupType: can be either audio, video, image or text
    :return: a dictionary with all the necessary data which is needed for analysis in the comparison module
    """
    dataDict = {}

    if(sourceType == 'video'):
        dataDict['sourceImages'] = videoToFrames(sourcePath,sourceFlag=True)
        dataDict['sourceAudio'] = video2audio(sourcePath)
        dataDict['sourceAudioText'] = audioToText(dataDict['sourceAudio'])
        dataDict['sourceImageText'] = imageToText(dataDict['sourceImages'])

    elif(sourceType == 'image'):
        imageList = []
        imageList.append(sourcePath)
        dataDict['sourceImageText'] = imageToText(imageList)

    elif (sourceType == 'audio'):
        dataDict['sourceAudioText'] = audioToText(sourcePath)

    elif (sourceType == 'text'):
        dataDict['sourceText'] = open(sourcePath,'r+')

    if (lookupType == 'video'):
        dataDict['lookupImages'] = videoToFrames(lookupPath,sourceFlag=False)
        dataDict['lookupAudio'] = video2audio(lookupPath)
        dataDict['lookupAudioText'] = audioToText(dataDict['lookupAudio'])
        dataDict['lookupImageText'] = imageToText(dataDict['lookupImages'])

    elif (lookupType == 'image'):
        imageList = []
        imageList.append(lookupPath)
        dataDict['lookupImageText'] = imageToText(imageList)

    elif (lookupType == 'audio'):
        dataDict['lookupAudioText'] = audioToText(lookupPath)

    elif (lookupType == 'text'):
        dataDict['lookupText'] = open(lookupPath,'r+')

    return dataDict

# if __name__ == '__main__':
#     video2audio(r'D:\Personal\SmartIT\data\dnc-2004-speech.mp4')

