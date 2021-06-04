from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import moviepy.editor as mp
import speech_recognition as sr
import spacy
import os
r = sr.Recognizer()
class integratedvideo2video:
    def imagecompare(self,image1,image2):
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
        s = ssim(original, contrast)
        print('mse', m, 'sse', s)
        return m,s

    def video2audio2text(self):
        video = r'dnc-2004-speech.mp4'  # Path where the videos are located
        clip = mp.VideoFileClip(video).subclip(0, 20)
        clip.audio.write_audiofile("theaudio.wav")
        r = sr.Recognizer()

        hellow = sr.AudioFile('theaudio.wav')
        with hellow as source:
            audio = r.record(source)
        try:
            s = r.recognize_google(audio)
            print("Text: " + s)
            file = open('data.txt', 'w')
            file.write(s)
            file.close()
        except Exception as e:
            print("Exception: " + str(e))

    def text2text(self,s1,s2):
        nlp = spacy.load('en_core_web_lg')
        search_doc = nlp(s1)
        main_doc = nlp(s2)
        print(main_doc.similarity(search_doc))
    def Patternmatching(self,source,destination):
        for file1 in glob.glob1(source,"*.jpg"):
            count = 0
            flag = False
            for file2 in glob.glob1(destination,"*.jpg"):
                if flag == True:
                    continue
                print (file1,file2)
                score = self.imagecompare(source+os.sep+file1,destination+os.sep+file2)
                meansquareerror = score[0]
                structuralsimilarity = score[1]
                if meansquareerror == 0.0 and structuralsimilarity == 1.0:
                    count +=1
                    flag = True
        return count





if __name__ == '__main__':
    int = integratedvideo2video()
    # print(type(int.imagecompare('comparison.jpg','comparison.jpg')))
    # a = int.imagecompare('comparison.jpg','comparison.jpg')
    # print(a[0])
    int.Patternmatching(r'C:\Users\kpmg\Desktop\fwdaudiotoaudiocomparision\data',r'C:\Users\kpmg\Desktop\fwdaudiotoaudiocomparision\data - Copy')