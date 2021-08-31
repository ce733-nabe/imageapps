from django.db import models
import numpy as np
import keras,sys
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io,base64

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import tensorflow as tf
import time
# Create your models here.
'''
#graph=tf.get_default_graph()
graph = tf.compat.v1.get_default_graph 
class Photo(models.Model):
    image=models.ImageField(upload_to="photos")

    IMAGE_SIZE=100#画像サイズ
    MODEL_PATH="./imageai/ml_models/model-1.h5"
    #imagename=[自分で作ったデータラベル]
    imagename=['test']
    image_len=len(imagename)

    def predict(self):
        model=None
        global graph#毎回同じモデルのセッションに投入して推論可能にする。
        with graph.as_default():
            model=load_model(self.MODEL_PATH)

            img_data=self.image.read()
            img_bin=io.BytesIO(img_data)

            image=Image.open(img_bin)
            image=image.convert("RGB")
            image=image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE))
            data=np.asarray(image)/255.0
            X=[]
            X.append(data)
            X=np.array(X)

            result=model.predict([X])[0]
            predicted=result.argmax()
            percentage=int(result[predicted]*100)

            return self.imagename[predicted],percentage
    def image_src(self):
        with self.image.open() as img:
            base64_img=base64.b64encode(img.read()).decode()

            return "data:"+img.file.content_type+";base64,"+base64_img
'''
class Photo(models.Model):
    image=models.ImageField(upload_to="photos")
    print('image:{}'.format(image))
    imagename=['test']

    def predict(self):
        model = tf.keras.applications.EfficientNetB0(weights='imagenet')

        width = 224
        height = 224

        img_data=self.image.read()
        img_bin=io.BytesIO(img_data)
        image=Image.open(img_bin)
        image=image.convert("RGB")
        image=image.resize((width,height))
        data=np.asarray(image)/255.0
        X=[]
        X.append(data)
        X=np.array(X)

        y = model.predict(X)
    
        yy = decode_predictions(y, top=1)
       
        yyy = yy[0][0]
      
        return yyy[1],int(yyy[2]*100)

    def image_src(self):
        with self.image.open() as img:
            base64_img=base64.b64encode(img.read()).decode()

            return "data:"+img.file.content_type+";base64,"+base64_img