from turtle import up
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import streamlit as st
happy=""
sad=""
neutral=""
angry=""
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
#uploaded_file = st.file_uploader("Choose a file",type=['mp4'])
def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Image Loaded")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('model.h5')
picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
    save_uploadedfile(picture)
    img=cv2.imread(os.path.join("tempDir",picture.name))
    labels = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            #cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            em=label
            #print(label)
            st.write('Emotion:')
            st.write(label)
              
    
# if uploaded_file is not None:
#     video_bytes= uploaded_file.getvalue()
#     st.video(video_bytes)
#     save_uploadedfile(uploaded_file)
#     face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     classifier =load_model('model.h5')
#     emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
#     cap = cv2.VideoCapture("/tempDir/"+uploaded_file.name)
    



