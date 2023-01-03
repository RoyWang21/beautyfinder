from PIL import Image
import pathlib
import cv2
import numpy as np
from split_image import split_image, reverse_split

imgdir_path = pathlib.Path('.')
#('./data')
file_list = sorted([str(path) for path in imgdir_path.glob('*.JPG')])
file_list = [fn.split('.')[0] for fn in file_list]
print(file_list)

# file_list = ['IMG_3170']

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

for file in file_list:
    face_found = False
    print(file)
    img = cv2.imread(file+'.JPG')
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 
        # img,
        1.0,
        # (w, h),
        (300,300),
        (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            face_found = True
            print('--face detected!')
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            img_box = cv2.rectangle(img, (x-50, y-50), (x1+50, y1+50), (0, 0, 255), 2)
          
    if face_found:
    # img_box = cv2.rectangle(img, (0, 0), (100, 100), (0, 0, 255), 2)
        cv2.startWindowThread()
        cv2.namedWindow("Image_with_face")
        cv2.imshow('Image_with_face', img_box) 
        cv2.waitKey(800)
        cv2.destroyAllWindows()
        
        # Cropping an image
        cropped_image = img[y-50:y1+50, x-50:x1+50]
        # Display cropped image
        cv2.imshow("cropped", cropped_image)
        # Save the cropped image
        cv2.imwrite(f"Cropped_{file}.jpg", cropped_image)
        
    else:
        print(f'!!! -- face not detected for {file}')