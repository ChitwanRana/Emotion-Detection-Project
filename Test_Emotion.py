import numpy
import cv2
from tensorflow.keras.models import model_from_json # type: ignore


emotion_dict={0:'Angry',1:'Disgusted',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprised'}


#load json and create model
json_file=open('model/emotion_model.json','r')
loaded_model_json=json_file.read()
json_file.close()
emotion_model=model_from_json(loaded_model_json)

#load weights into new model 
emotion_model.load_weights("model/emotion_model.h5")
print("loaded model from disk")


