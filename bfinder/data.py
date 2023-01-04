import subprocess
from PIL import Image
import pathlib
import cv2
import numpy as np
from split_image import split_image, reverse_split
import os

def get_positive_labels(data_dir):
    """
    Get a set of files which were labelled as positive examples

    Parameters
    ----------
    data_dir : STR
        Directory of raw data pictures.

    Returns
    -------
    SET
        positive example flienames.

    """
    tag_clrs = ["Red", "Yellow", "Purple", "Blue", "Green", "Orange", "Grey"]
    
    file_list = []
    for cl in tag_clrs:
        query_pos = ["mdfind", "-onlyin",  f"{data_dir}", f"kMDItemUserTags == {cl}"]
        result = subprocess.run(query_pos, stdout=subprocess.PIPE, text=True)
        result = result.stdout.split('\n')[:-1]
        
        for fn in result:
            file_list.append(fn.split('/')[-1])
            
    return set(file_list)


def detect_face(img, model):

    face_found = False
    head_box = None
    h, w = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 
        # img,
        1.0,
        # (w, h),
        (300,300),
        (104.0, 117.0, 123.0))
    model.setInput(blob)
    faces = model.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            face_found = True
            print('--face detected!')
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            head_box = box.astype("int")
            break
    return face_found, head_box
        

def get_head_crops(data_dir, visualize=False):
    output_path = "data/crop_head/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(output_path)
       print(f"The new directory {output_path} is created!")
   
    imgdir_path = pathlib.Path(data_dir)
    file_list = sorted([str(path) for path in imgdir_path.glob('*.JPG')])
    file_list = [fn.split('.')[0] for fn in file_list]
    print(file_list)
    
    modelFile = "pretrained_models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "pretrained_models/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    
    for file in file_list:
        print(file)
        img = cv2.imread(file+'.JPG')
        face_found, head_box = detect_face(img, net)
              
        if face_found:
            x, y, x1, y1 = head_box
            # img_box = cv2.rectangle(img, (0, 0), (100, 100), (0, 0, 255), 2)
            img_box = cv2.rectangle(img, (x-50, y-50), (x1+50, y1+50), (0, 0, 255), 2)
            cropped_image = img[y-50:y1+50, x-50:x1+50]
            if visualize:
                cv2.startWindowThread()
                cv2.namedWindow("Image_with_face")
                cv2.imshow('Image_with_face', img_box) 
                cv2.waitKey(400)
                cv2.destroyAllWindows()
                cv2.imshow("cropped", cropped_image)
                
            # Save the cropped image
            fn = file.split('/')[-1]
            cv2.imwrite(output_path+f"h_{fn}.jpg", cropped_image)
            
        else:
            print(f'!!! -- face not detected for {file}')
        
        
if __name__ == '__main__':
    data_dir = 'data/.'
    print(get_positive_labels(data_dir))
    get_head_crops(data_dir)