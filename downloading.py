import boto3
import botocore
import os

BUCKET_NAME = 'lpl-demo' # source folder input

PREFIX = 'input/'   # inpput s3 folder name
OUTPUT_STORAGE = 'local_input_folder/'   # Output storage in

#s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

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
    return lst_filame[1:]

def download_files(file_list, OUTPUT_STORAGE):
    for x in range(len(file_list)):
        try:
            print("Downloading ", PREFIX+file_list[x])
            print("Downloading file name ", file_list[x])
            s3.download_file(BUCKET_NAME, PREFIX+file_list[x], OUTPUT_STORAGE+file_list[x])

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    
    
        copy_source = {
        'Bucket': BUCKET_NAME,
        'Key': PREFIX+file_list[x]
        }
        fle = 'input_archive/' + file_list[x]
        s3.copy(copy_source, BUCKET_NAME, fle)
        s3.delete_object(Bucket=BUCKET_NAME, Key=PREFIX+file_list[x])
        print("Deleted ", file_list[x])

#listing files
file_list = ListFiles()
#downloading files
download_files(file_list, OUTPUT_STORAGE)