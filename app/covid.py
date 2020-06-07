import cv2
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from os.path import isfile
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

class Covid:
    def __init__(self, IMAGE_PATH, MODEL_PATH):
        self.IMAGE_PATH = IMAGE_PATH
        self.MODEL_PATH = MODEL_PATH

    def get_class_activation_map(self,predict,model,img) :
        if predict==1:
            return
        np.seterr(divide='ignore', invalid='ignore')
        img=np.expand_dims(img, axis=0)
        img= np.array(img) / 255.0
        target_class = predict
        last_conv = model.get_layer('block5_conv3')
        grads =K.gradients(model.output[:,target_class],last_conv.output)[0]
        pooled_grads = K.mean(grads,axis=(0,1,2))
        iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
        pooled_grads_value,conv_layer_output = iterate([img])

        for i in range(512):
            conv_layer_output[:,:,i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output,axis=-1)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x,y] = np.max(heatmap[x,y],0)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        upsample = cv2.resize(heatmap, (880,882))
        img_gray= cv2.resize(img_gray,(880,882))
        path=self.IMAGE_PATH.split('data')
        path_heatmap='heatmap'+path[1]
        output_path_gradcam = path[0]+'static/'+path_heatmap
        plt.imsave(output_path_gradcam,(upsample * img_gray))

    def covid_predict(self):
        # Before prediction
        K.clear_session()
        model = load_model(self.MODEL_PATH)
        # Loading and preprocessing image
        img = load_img(self.IMAGE_PATH, color_mode='rgb', target_size=(
            224, 224))
        image = img_to_array(img)
        img = np.expand_dims(image, axis=0)
        data = np.array(img) / 255.0
        # Get predictions for image
        prediction = ['Positive', 'Negative']
        preds = model.predict(data)
        predict = np.argmax(preds)
        probability = "{:.2f} %".format((preds[0][predict])*100)
        print(predict)
        self.get_class_activation_map(predict,model,image)
        return {'prediction':prediction[predict],'probability':probability}

