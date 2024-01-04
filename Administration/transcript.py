import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append('nltk_data')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cv2
from fer import FER
import math

import requests

API_KEY = "623cfea0aba24d8f981195bbc20d48e0"

def upload_and_transcribe_audio(video_file_path):
    filename = video_file_path
    transcript = ""
    try:
        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        headers = {'authorization': API_KEY}
        
        # Upload the audio file
        with open(video_file_path, 'rb') as vf:
            response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=read_file(filename))
        json_str1 = response.json()

        # Create a transcription job
        endpoint = "https://api.assemblyai.com/v2/transcript"
        json_data = {
            "audio_url": json_str1["upload_url"]
        }
        response = requests.post(endpoint, json=json_data, headers=headers)
        json_str2 = response.json()

        # Get the transcript
        endpoint = "https://api.assemblyai.com/v2/transcript/" + json_str2["id"]
        response = requests.get(endpoint, headers=headers)
        json_str3 = response.json()

        while json_str3["status"] != "completed":
            response = requests.get(endpoint, headers=headers)
            json_str3 = response.json()
        
        transcript = json_str3["text"]
    except Exception as e:
        print(f"An error occurred: {e}")

    return transcript




def FindAcc(S1, S2):
    try:
        X = S1.lower()
        Y = S2.lower()

        S1 = re.split(r'[ ,.!;"()]', X)
        S2 = re.split(r'[ ,.!;"()]', Y)

        S1.sort()
        S2.sort()

        Positive = 0
        Negative = 0

        if len(S1) == 1:
            if S1[0] in S2:
                AccPer = 100
            else:
                AccPer = 0
            return AccPer

        if len(S2) == 1:
            S2.append(".")

        for i in S1:
            if i == "":
                continue

            if i in S2:
                Positive += 1
            else:
                Negative += 1

        Total = Positive + Negative

        AccPer = (Positive * 100) / Total

        if Negative < 5:
            X_list = word_tokenize(X)
            Y_list = word_tokenize(Y)

            sw = stopwords.words("english")
            l1 = []
            l2 = []

            X_set = {w for w in X_list if not w in sw}
            Y_set = {w for w in Y_list if not w in sw}

            rvector = X_set.union(Y_set)
            for w in rvector:
                if w in X_set:
                    l1.append(1)  # create a vector
                else:
                    l1.append(0)
                if w in Y_set:
                    l2.append(1)
                else:
                    l2.append(0)
            c = 0

            for i in range(len(rvector)):
                c += l1[i] * l2[i]
            cosine = c / float((sum(l1) * sum(l2)) ** 0.5)

            if min(AccPer, (cosine * 100)) < 40:
                AccPer = min(AccPer, cosine)
            else:
                AccPer = max(AccPer, cosine)

        return AccPer
    
    except:
        return 0  # or any other value that indicates an error


def analyze_video_emotions(video_filename):
    vid = cv2.VideoCapture(video_filename)
    fps = int(vid.get(cv2.CAP_PROP_FPS) / 3)  # Process every third frame
    emotion_detector = FER()
    n = 0
    i = 0
    sad1 = fear1 = happy1 = angry1 = surprise1 = disgust1 = neutral1 = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if n % fps == 0:
            attri = emotion_detector.detect_emotions(frame)
            print(attri)
            if len(attri) > 0:
                sad1 += attri[0]["emotions"]['sad']
                fear1 += attri[0]["emotions"]['fear']
                happy1 += attri[0]["emotions"]['happy']
                angry1 += attri[0]["emotions"]['angry']
                surprise1 += attri[0]["emotions"]['surprise']
                disgust1 += attri[0]["emotions"]['disgust']
                neutral1 += attri[0]["emotions"]['neutral']
                i += 1
        n += 1
    vid.release()

    total = sad1 + fear1 + happy1 + angry1 + surprise1 + disgust1 + neutral1
    
    if total == 0:
        confidence = 0
        nervousness = 0
    else:
        confidence = ((happy1 + surprise1) / total) * 100
        nervousness = ((sad1 + fear1 + disgust1) / total) * 100

        if confidence % 1 > 0.4:
            confidence = math.ceil(confidence)
        else:
            confidence = math.floor(confidence)

        if nervousness % 1 > 0.4:
            nervousness = math.ceil(nervousness)
        else:
            nervousness = math.floor(nervousness)

        neutral1 = 100 - (confidence + nervousness)

    return confidence, nervousness, neutral1

