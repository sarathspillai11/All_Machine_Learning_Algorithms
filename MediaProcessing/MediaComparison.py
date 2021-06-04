from MediaProcessing.MediaPreparator import mediaPreparation as media_prep
import spacy
from skimage import measure
import numpy as np
import cv2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import glob
import os
#import azure.cognitiveservices.speech as speechsdk
sourcePath = ''


def text2text(s1, s2):
    nlp = spacy.load('en_core_web_lg')
    search_doc = nlp(s1)
    main_doc = nlp(s2)
    print(main_doc.similarity(search_doc))
    return (main_doc.similarity(search_doc))


def imagecompare(image1, image2):
    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    original = cv2.imread(image1)
    contrast = cv2.imread(image2)
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    m = mse(original, contrast)
    s = measure.compare_ssim(original, contrast)
    print('mse', m, 'sse', s)
    return m, s


def Patternmatching(source,destination,continuty):
    count = 0
    continuecount = 0
    for file1 in glob.glob1(source,"*.jpg"):
        flag = False
        for file2 in glob.glob1(destination,"*.jpg"):
            if flag == True:
                continue
            score = imagecompare(source+os.sep+file1,destination+os.sep+file2)
            meansquareerror = score[0]
            structuralsimilarity = score[1]
            if continuty:
                if meansquareerror == 0.0 and structuralsimilarity == 1.0:
                    continuecount +=1
                    flag = True
                else:
                    continuecount = 0
            else:
                if meansquareerror == 0.0 and structuralsimilarity == 1.0:
                     count +=1
                     flag = True
    return count,continuecount

def audio2textapi(audiofile):
    speech_key, service_region = "I3_Xc6FKJFTakajfVotqIQ4bqVa53Tw9XYyuMjfF_Iw", "westus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Creates an audio configuration that points to an audio file.
    # Replace with your own audio filename.
    audio_filename = audiofile
    audio_input = speechsdk.AudioConfig(filename=audio_filename)
    
    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    
    print("Recognizing first result...")
    
    
    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed.  The task returns the recognition text as result. 
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query. 
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()
    
    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    
def findSentiment(text):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return score['compound']

def mediaCompare(sourcePath,lookupPath,sourceType,lookupType,compareTextorImage,textFromImageFlag,analysisTypes):


    dataDict = media_prep(sourcePath,sourceType,lookupPath,lookupType)

    if(dataDict == {}):
        print('Data Preparation Failed ..')
        raise Exception

    for analysisType in analysisTypes:
        if(analysisType=='similarity'):
            if(compareTextorImage == 'text'):
                if(sourceType != 'text'):
                    if(textFromImageFlag == True):
                        out = text2text(dataDict['sourceImageText'],dataDict['lookupImageText'])
                    else:
                        out = text2text(dataDict['sourceAudioText'], dataDict['lookupAudioText'])
                else:
                    text2text(dataDict['sourceText'], dataDict['lookupImageText'])
            else:
                print('Please opt for Pattern matching to compare images')
                out = 0
                return out
            print('out :',out)

        elif(analysisType == 'sentiment'):
            if (textFromImageFlag == True):
                fullTextFromImage = '.'.join([text for text in dataDict['sourceImageText']])
                out = findSentiment(fullTextFromImage)
            else:
                print(dataDict['sourceAudioText'])
                out = findSentiment(dataDict['sourceAudioText'])


        elif (analysisType == 'pattern'):
            out = Patternmatching(os.getcwd()+os.sep+'sourceImages',os.getcwd()+os.sep+'lookupImages')


        print('out :',out)
        return out




if __name__ == '__main__':
    mediaCompare(sourcePath=r'D:\Personal\SmartIT\data\ClearEnglishSpeech.mp4',lookupPath=r'D:\Personal\SmartIT\data\ClearEnglishSpeech.mp4',sourceType='video',lookupType='video',compareTextorImage='image',textFromImageFlag=True,analysisTypes=['pattern'])





