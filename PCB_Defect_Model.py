import pandas as pd
import numpy as np
import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class Model_load:
    def __init__(self, model_pt_path):
        self.load_model(model_pt_path)

    def load_model(self, model_pt_path):
        # load model

        best_weight_path = model_pt_path
        self.model = YOLO(model = best_weight_path)


    def preprocess_image_path(self, img_path, target_size=(256, 256)):
        """
        이미지 파일 경로를 받아서 모델 입력에 맞게 전처리
        """
        # 이미지 불러오기
        img = cv2.imread(img_path)

        # 이미지 정규화
        # img_array = img_array.astype('float32') / 255.
        return img_array

    def preprocess_image_list(self, img_list, target_size=(256, 256)):
        """
        이미지를 ndarray 형태로 바꾼 뒤 정규화 처리
        """
        # 리스트를 numpy 배열로 변환, dtype을 uint8로 최적화
        img_array = np.array(img_list, dtype='uint8')

        # 이미지 정규화
        # img_array = img_array.astype('float32') / 255.0

        return img_array

    def predict_defect(self, input_img):
        """
        하나의 이미지의 양불을 예측하고 해당하는 label과 이미지 ndarray를 dictionary 형태로 반환
        """
        
        # 이미지 양불판정
        self.model.predict(input_img, save = True)
        
        result = self.model.predict(input_img)[0]

        # image = result.plot()
        
        # label_class로 되돌리기
        defect_names = []
        
        model_names = self.model.names
        
        detection = self.model(input_img)[0]
        
        boxinfos = detection.boxes.data.tolist()
        
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
        threshold = 0.4
        
        for data in boxinfos :
            x1,y1,x2,y2 = map(int,data[:4])
            confidence_score = round(data[4],2)
            classid = int(data[5])
            name = model_names[classid]
            defect_names.append(name)
        
        defect_name_set = set(defect_names)
        defect_name_list = list(defect_name_set)
        
        res_dict = {
            "defect_name_list" : defect_name_list,
            "image" : boxinfos
}

        return res_dict