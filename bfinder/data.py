import subprocess
from PIL import Image
import pathlib
import cv2
import numpy as np
from split_image import split_image, reverse_split
import os
import shutil

class DataETL:
    def __init__(self, data_dir, output_path):
        self.data_dir = data_dir
        self.output_path = output_path
        self.modelFile = "pretrained_models/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "pretrained_models/deploy.prototxt.txt"
        self.all_examples = set()
        self.pos_examples = set()
        self.train_examples = set()
        self.val_examples = set()

    def get_positive_labels(self):
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
            query_pos = ["mdfind", "-onlyin",  f"{self.data_dir}", f"kMDItemUserTags == {cl}"]
            result = subprocess.run(query_pos, stdout=subprocess.PIPE, text=True)
            result = result.stdout.split('\n')[:-1]
            
            for fn in result:
                file_list.append(os.path.basename(fn))
                
        self.pos_examples = set(file_list)
    
    @staticmethod
    def _detect_face(img, model):
    
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
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                head_box = box.astype("int")
                break
        return face_found, head_box
            
    def _load_face_model(self):
        net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        return net
    
    @staticmethod
    def _util_creat_folder(path):
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(path)
           print(f"The new directory {path} is created!")
           
    @staticmethod
    def _copy_files(file_list, tgt_dir):
        try:
            for src_file_path in file_list:
                file_name = os.path.basename(src_file_path)
                tgt_file_path = os.path.join(tgt_dir, file_name)
                if os.path.isfile(src_file_path):
                    shutil.copy(src_file_path, tgt_file_path)
            print('copy files successfully')
        except Exception as e:
            print('error copy files! Due to:',e)
        
    def get_head_crops(self, visualize=False):
        
        self._util_creat_folder(self.output_path)
       
        imgdir_path = pathlib.Path(self.data_dir)
        file_list = sorted([str(path) for path in imgdir_path.glob('*.JPG')])
        file_list = [fn.split('.')[0] for fn in file_list]
        print(file_list)
        
        net = self._load_face_model()
        
        for file in file_list:
            img = cv2.imread(file+'.JPG')
            face_found, head_box = self._detect_face(img, net)
                  
            if face_found:
                print(f'--face detected for {file}!')
                x, y, x1, y1 = head_box
                # img_box = cv2.rectangle(img, (0, 0), (100, 100), (0, 0, 255), 2)
                img_box = cv2.rectangle(img, (x-50, y-50), (x1+50, y1+50), (0, 0, 255), 2)
                cropped_image = img[y-50:y1+50, x-50:x1+50]
                    
                # Save the cropped image
                fn = output_path + f"h_{os.path.basename(file)}.jpg"
                cv2.imwrite(fn, cropped_image)
                self.all_examples.add(fn)
                
                if visualize:
                    cv2.startWindowThread()
                    cv2.namedWindow("Image_with_face")
                    cv2.imshow('Image_with_face', img_box) 
                    cv2.waitKey(400)
                    cv2.destroyAllWindows()
                    cv2.imshow("cropped", cropped_image)
                
            else:
                print(f'!!! -- face not detected for {file}')
            
    
    def get_data_splits(self, train_size=0.8):
        self.train_path = os.path.join(self.data_dir, 'train')
        self.val_path = os.path.join(self.data_dir, 'val')
        
        self._util_creat_folder(self.train_path)
        self._util_creat_folder(self.val_path)
        # get into train and val folders
        example_list = list(self.all_examples)
        train_examples = np.random.choice(example_list,
                         int(train_size * len(example_list)),
                         replace=False)
        self.train_examples = set(train_examples)
        self.val_examples = self.all_examples - self.train_examples
        # copy files into folders
        self._copy_files(self.train_examples, self.train_path)
        self._copy_files(self.val_examples, self.val_path)


if __name__ == '__main__':
    data_dir = 'data/'
    output_path = "data/crop_head/"
    data_etl = DataETL(data_dir, output_path)
    data_etl.get_positive_labels()
    print('pos_example',data_etl.pos_examples)
    data_etl.get_head_crops()
    data_etl.get_data_splits()
    print('all_example',data_etl.all_examples)
    print('train_example',data_etl.train_examples)
    print('val_example',data_etl.val_examples)