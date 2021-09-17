import os
import time
import xgboost as xgb
from keras.applications.vgg16 import VGG16
import numpy as np 
#import matplotlib.pyplot as plt
import cv2
import shutil
import boto3
import botocore



########################################


# searching for images
def file_list_pred(path):
    file_names = []
    files = os.listdir(path)
    print("loading image files in",path[:-1])
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = path + file
            #print(img_path)
            file_names.append(img_path)
    #print(file_names)
    return file_names


# prediction part
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
    #print("The prediction is: ", result)
    final = "The prediction for the image(" + str(img_path) + ") is: " + result
    #print("The prediction for this image is: ", result)
    print(final)
    #with open('/tmp/gg-amazing-forestfire.log', 'a') as f: print(final, file=f)
    if prediction == 0:
        #print("fire",str(img_path))
        shutil.move(img_path, "local_output_folder/fire/" + img_path[img_path.index('/')+1:])
    else:
        #print("In else loop, no fire")
        shutil.move(img_path, "local_output_folder/nofire/" + img_path[img_path.index('/')+1:])
    return final,prediction


# for images
def image_loop(image_list):
    for x in image_list:
        #print(x)
        prediction(model_name,x)








# Listing files within s3 bucket /input
def ListFiles():
    lst_filame = []
    lst_pthname = []
    # List files in specific S3 URL
    response = s3.list_objects(Bucket=BUCKET_NAME, Prefix=PREFIX)
    for content in response.get('Contents', []):
        #print(content.get('Key'))
        b = content.get('Key')
        b = b[b.index('/')+1:]
        lst_filame.append(b)
        lst_pthname.append(content.get('Key'))
    return lst_filame[1:],lst_pthname  


# Downloading files to local
def download_files(file_list, path_list, OUTPUT_STORAGE):
    for x in range(len(file_list)):
        try:
            print("Downloading ", path_list[x])
            s3.download_file(BUCKET_NAME, path_list[x], OUTPUT_STORAGE+file_list[x])

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
        


#local output fire upload
local_input_firefilepath = 'local_output_folder/fire' 
def Uploading(local_input_firefilepath):
    for file in os.listdir(local_input_firefilepath):
        if not file.startswith('~'):
            try:
                print('Uploading File {0} ....'.format(file))
                s3.upload_file(
                    os.path.join(local_input_firefilepath, file),
                    BUCKET_NAME,
                    "output/fire" + file
                )


            except Exception as e:
                print(e)
                

#local output fire upload
local_input_nofirefilepath = 'local_output_folder/nofire' 
def Uploading(local_input_nofirefilepath):
    for file in os.listdir(local_input_nofirefilepath):
        if not file.startswith('~'):
            try:
                print('Uploading File {0} ....'.format(file))
                s3.upload_file(
                    os.path.join(local_input_nofirefilepath, file),
                    BUCKET_NAME,
                    "output/fire" + file
                )


            except Exception as e:
                print(e)



BUCKET_NAME = 'lpl-demo' # source folder input
PREFIX = 'input/'   # input s3 folder name
OUTPUT_STORAGE = 'local_input_folder/'   # Output storage in




#Load model wothout classifier/fully connected layers
SIZE = 256  #Resize images
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

path = "local_input_folder/"
model_name = "lpl_forestfire_model.tflite"




# main program starts here   
i = 0
while i<=10:
    os.system("python downloading.py")
    out_file = file_list_pred(path)
    if len(out_file) >= 1:
        #print(len(out_file))
        image_loop(out_file)
        # Uploading fire
        print("Uploading files!!!!!!!!!!!!!!!!!!!!!!")
        Uploading(local_input_nofirefilepath)

        # Uploading nofire
        Uploading(local_input_nofirefilepath)
    
    else:
        print("No images in the folder!!.")

    time.sleep(3)