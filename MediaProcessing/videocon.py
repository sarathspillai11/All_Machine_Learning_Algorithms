# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:07:08 2019

@author: vamsi
"""
curpath = r'C:\Users\vamsi\OneDrive\Documents\fwdaudiotoaudiocomparision'
import os
os.chdir(curpath)
video = r'dnc-2004-speech.mp4'  # Path where the videos are located
import moviepy.editor as mp
clip = mp.VideoFileClip(video).subclip(0,20)
clip.audio.write_audiofile("theaudio.wav")

import speech_recognition as sr
r = sr.Recognizer()

hellow=sr.AudioFile('theaudio.wav')
with hellow as source:
    audio = r.record(source)
try:
    s = r.recognize_google(audio)
    print("Text: "+s)
    file = open('data.txt','w')
    file.write(s)
    file.close()
except Exception as e:
    print("Exception: "+str(e))

import spacy
nlp = spacy.load('en_core_web_lg')

search_doc = nlp(s)

main_doc = nlp(s)

print(main_doc.similarity(search_doc))