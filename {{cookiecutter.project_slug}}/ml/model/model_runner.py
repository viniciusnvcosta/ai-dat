import torch
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
from app.core.config import settings
import tensorflow as tf

load_dotenv()

class YoloDetector():

    def __init__(self, model):
        self.model = model
        self.img_size = settings.IMG_SIZE
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

    def inference(self, input_img):
        ## BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        ## Opencv to PIL
        input_img = Image.fromarray(input_img)
        detection_results = self.model.predict(source=input_img, save_conf=True, verbose=False, imgsz=self.img_size)
        dict_results = {"names":{}, "bboxes":[], "classes":[], "confs":[]}
        dict_results["names"] = detection_results[0].names
        dict_results["bboxes"] = detection_results[0].boxes.xywh.tolist()
        dict_results["classes"] = detection_results[0].boxes.cls.tolist()
        dict_results["confs"] = detection_results[0].boxes.conf.tolist()

        return dict_results

class YoloClassifier():

    def __init__(self, model):
        self.model = model
        self.img_size = settings.IMG_SIZE
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

    def inference(self, input_img):
        dict_results = {'names':{}, 'softs':[], 'pred':[]}
        results = self.model.predict(source=input_img,save_conf=True,verbose=False, imgsz=self.img_size)
        dict_results['names'] = results[0].names
        for result in results:
            dict_results['softs'].append(result.probs.data.tolist())
            dict_results['pred'].append([result.probs.top1])
        return dict_results

class TensorflowClassifier():
    def __init__(self, model):
        self.model = model
        self.img_size = settings.IMG_SIZE # Image size needs to match the model's input size, change in the .env if necessary

    def inference(self, input_img):
        # BGR to RGB
        input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # Opencv to PIL
        input_img_pil = Image.fromarray(input_img_rgb)
        # Preprocessing as required for the model
        input_img_resized = input_img_pil.resize((self.img_size, self.img_size))
        input_img_array = tf.keras.preprocessing.image.img_to_array(input_img_resized)
        input_img_normalized = input_img_array / 255.0  # Normalization

        # Adding batch dimension
        input_img_expanded = tf.expand_dims(input_img_normalized, axis=0)

        # Doing inference
        predictions = self.model.predict(input_img_expanded)
        # Post-processing the inference result
        output_list = []

        for prediction in predictions:
            # Getting the index of the class with the highest probability
            predicted_class_index = np.argmax(prediction)

            # Getting the classification score of the predicted class
            classification_score = prediction[predicted_class_index]

            # Adding to the output list
            output_list.append({
                "class_name": predicted_class_index,
                "classification_score": classification_score
            })

        return output_list