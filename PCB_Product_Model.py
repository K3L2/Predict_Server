import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

class Model_load:
    def __init__(self,model_h5_path, class_json_path):
        self.load_model(model_h5_path, class_json_path)

    def load_model(self,model_h5_path, class_json_path):
        '''h5형식 모델과 json형식 클래스 로드'''
        self.model = load_model(model_h5_path)
        with open(class_json_path, 'r') as f:
            self.class_dict = json.load(f)
            self.reverse_class_dict = dict((v, k) for k, v in self.class_dict.items())


    def preprocess_image_path(self, img_path, target_size=(256, 256)):
        """이미지 파일 경로를 받아서 모델 입력에 맞게 전처리"""
        # 이미지 불러오기
        img = cv2.imread(img_path)
        # 이미지 크기 조정
        img = cv2.resize(img, target_size)
        # BGR에서 RGB로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 이미지를 배열로 변환하고 차원 확장 (1, 256, 256, 3)
        img_array = np.expand_dims(img, axis=0)
        # 이미지 정규화
        img_array = img_array.astype('float32') / 255.
        return img_array

    def preprocess_image_list(self, img_list, target_size=(256, 256)):
        """
        사이즈가 (W, H, 3)인 리스트 데이터를 (1, 256, 256, 3) 형태의 np.ndarray로 변환
        """
        # 리스트를 numpy 배열로 변환, dtype을 uint8로 최적화
        img_array = np.array(img_list, dtype='uint8')

        # 이미지 크기 조정 (256, 256)
        img_array = cv2.resize(img_array, target_size)

        # 이미지 차원 확장 (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 이미지 정규화
        img_array = img_array.astype('float32') / 255.0

        return img_array

    def predict_image(self, is_path: bool, image):
        """하나의 이미지를 예측하고 원래 라벨을 출력
        is_path : True -> image : path_str
        is_path : False -> image : list (H,W,3) """
        if is_path:
            img_array = self.preprocess_image_path(image)
        else:
            img_array = self.preprocess_image_list(image)
        # 이미지 라벨 예측
        prediction = self.model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # label_class로 되돌리기
        predicted_label = self.reverse_class_dict[predicted_class_index]

        return str(predicted_label).zfill(2)