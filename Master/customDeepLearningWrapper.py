import Master.MasterAlgorithm as master
import Deep_Learning_Usecases.ageGroupDetectionKerasMaster.detectGroup as ageDetect
import Deep_Learning_Usecases.ageGroupDetectionKerasMaster.detectGroupWebcam as ageDetectWebcam
import Deep_Learning_Usecases.ChildVsAdultDetectionKerasMaster.detectChildVsAdult as childAdult
import Deep_Learning_Usecases.ChildVsAdultDetectionKerasMaster.detectChildAdultWebcam as childAdultWebcam
import Deep_Learning_Usecases.genderDetectionKerasMaster.detectGender as genderDetect
import Deep_Learning_Usecases.genderDetectionKerasMaster.detectGenderWebcam as genderDetectWebcam
import Deep_Learning_Usecases.raceDetectionKerasMaster.detectRace as raceDetect
import Deep_Learning_Usecases.raceDetectionKerasMaster.detectRaceWebcam as raceDetectWebcam
import  Deep_Learning_Usecases.RealtimeEmotionDetectionMaster.videoTester as emotionDetect
import Deep_Learning_Usecases.RealtimeMotionDetection.MotionDetection as motionDetect

def deepLearningWrapper(attributeDictionary,wrapperModelCombinations):
    ticketId = attributeDictionary['ticketId']
    for useCase in wrapperModelCombinations:
        if(useCase=='ageGroupDetection'):
            ageDetect.detectAge()
        elif(useCase=='ageGroupDetectionWebcam'):
            ageDetectWebcam.detectAgeFromWebcam()

        elif(useCase=='childAdultDetection'):
            childAdult.childAdultDetect()
        elif (useCase == 'childAdultDetectionWebcam'):
            childAdultWebcam.childAdultDetectWebcam()

        elif (useCase == 'genderDetection'):
            genderDetect.genderDetector()
        elif (useCase == 'genderDetectionWebcam'):
            genderDetectWebcam.genderDetectWebcam()

        elif (useCase == 'raceDetection'):
            raceDetect.raceDetector()
        elif (useCase == 'raceDetectionWebcam'):
            raceDetectWebcam.raceDetectWebcam()

        elif (useCase == 'emotionDetection'):
            emotionDetect.videoTest()

        elif(useCase=='motionDetector'):
            motionDetect.motionDetector()
