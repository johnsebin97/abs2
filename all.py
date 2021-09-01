import os
import time
import xgboost as xgb
from keras.applications.vgg16 import VGG16
import numpy as np 
#import matplotlib.pyplot as plt
import cv2


#Load model wothout classifier/fully connected layers
SIZE = 256  #Resize images
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

path = "input_folder/"
model_name = "lpl_forestfire_model.tflite"

def file_list(path):
    file_names = []
    files = os.listdir(path)
    print("loading image files in",path[:-1])

    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = path + file
            #print(img_path)
            file_names.append(img_path)

    print(file_names)
    return file_names

def prediction(model_name,img_path):
    model2 = xgb.XGBClassifier()
    model2.load_model(model_name)
    
    #Image Processing
    img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, (SIZE, SIZE))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)


    #plt.imshow(img1)

    input_img = np.expand_dims(img1, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_feature=VGG_model.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction = model2.predict(input_img_features)[0] 
    #prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
    
    result = "positive fire" if prediction == 0 else "negative fire"
    print("The prediction is: ", result)
    final = "The prediction for the image(" + str(img_path) + ") is: " + result
    #print("The prediction for this image is: ", result)
    print(final)
    #with open('/tmp/gg-amazing-forestfire.log', 'a') as f: print(final, file=f)
    
    return final,prediction

def image_loop(image_list):
    for x in image_list:
        #print(x)
        prediction(model_name,x)
    
    
i = 0
while i<=10:
    print("TRUE")
    out_file = file_list(path)
    if len(out_file) >= 1:
        print("in for loop")
        print(len(out_file))
        image_loop(out_file)
    
    else:
        print("No images in the folder.")
    
    time.sleep(3)