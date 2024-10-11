from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as T
from cv2.typing import MatLike
from loguru import logger
from PIL import Image
from ultralytics import YOLO

from app.core.config import settings


class YoloDetector:
    """
    A class to handle YOLO model detection tasks, including preprocessing images, performing inference,
    and formatting detection results.
    Attributes:
        model (YOLO): The YOLO model used for detection.
        img_size (int): The size of the input image required by the model.
        conf_thres (float): The confidence threshold for detection.
        iou_thres (float): The Intersection Over Union (IOU) threshold for detection.
    Methods:
        __init__(model: YOLO):
            Initializes the YoloDetector with the given YOLO model and settings.
        preprocess_image(image: MatLike) -> torch.Tensor:
            Preprocesses the input image by converting it to RGB, resizing, and normalizing it.
        inference(input_img: MatLike) -> Dict:
            Performs inference on the input image and returns formatted detection results.
        _transform_coords(coords: List[int]) -> List[int]:
            Transforms bounding box coordinates from [x_min, y_min, x_max, y_max] to [x, y, width, height].
    """

    def __init__(self, model: YOLO):
        self.model = model
        # Image size needs to match the model's input size, change in the .env if necessary
        self.img_size = settings.IMAGE_SIZE
        self.conf_thres = settings.DETECTION_THRESHOLD
        self.iou_thres = settings.OVERLAP_THRESHOLD
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

    def preprocess_image(self, image: MatLike) -> torch.Tensor:
        """
        Preprocess the input image as needed.
        This step may include resizing, normalizing, or any custom preprocessing required.
        """

        # Convert image to RGB if necessary
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and pad the image to fit the required input size while maintaining aspect ratio
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
            ]
        )
        img = transform(image)
        img = torch.unsqueeze(img, 0)

        # Move the image to the specified device
        img = img.to(self.model.device)

        return img  # Return the preprocessed image as a tensor

    def inference(self, input_img: MatLike) -> Dict:
        """
        Perform inference on the input image and return YOLO formatted detection results.
        """
        logger.info("Inference started")
        img = self.preprocess_image(input_img)

        # Perform inference using the YOLO model
        results = self.model.predict(
            source=img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.img_size,
            save_conf=True,
            verbose=False,
        )[0]

        # Debug
        logger.debug(f"Raw Detection Results: {results}")

        # Prepare the results dictionary
        dict_results = {
            "names": results.names,
            "bboxes": [],
            "classes": [],
            "confs": [],
        }

        if results.boxes:
            logger.info("bboxes found")
            dict_results["bboxes"] = [
                # * add transformations if necessary
                box
                for box in results.boxes.xywh
            ]
            dict_results["classes"] = results.boxes.cls.tolist()
            dict_results["confs"] = results.boxes.conf.tolist()

        # Debug:
        logger.debug(f"Formatted Results: {dict_results}")

        logger.info("Inference completed")

        return dict_results

    def _transform_coords(self, coords: List[int]) -> List[int]:
        """
        Transform the bounding box coordinates from [x_min, y_min, x_max, y_max] to [x, y, width, height].

        note(V): This method is not used in the current implementation but can be used if necessary.
        """
        x_min, y_min, x_max, y_max = coords
        width = x_max - x_min
        height = y_max - y_min
        return [int(x_min), int(y_min), int(width), int(height)]


class YoloClassifier:

    def __init__(self, model: tf.keras.models.Model):
        self.model = model
        self.img_size = settings.IMAGE_SIZE
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

    def preprocess_image(self, image: MatLike) -> torch.Tensor:
        """
        Preprocess the input image as needed.
        This step may include resizing, normalizing, or any custom preprocessing required.
        """

        # Convert image to RGB if necessary
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and pad the image to fit the required input size while maintaining aspect ratio
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
            ]
        )
        img = transform(image)
        img = torch.unsqueeze(img, 0)

        # Move the image to the specified device
        img = img.to(self.model.device)

        return img  # Return the preprocessed image as a tensor

    def inference(self, input_img) -> Dict:
        """
        Perform inference on the input image and return YOLO formatted classification results.
        """
        logger.info("Inference started")
        img = self.preprocess_image(input_img)

        dict_results = {"names": {}, "softs": [], "pred": []}
        results = self.model.predict(
            source=img, save_conf=True, verbose=False, imgsz=self.img_size
        )
        dict_results["names"] = results[0].names
        for result in results:
            dict_results["softs"].append(result.probs.data.tolist())
            dict_results["pred"].append([result.probs.top1])
        return dict_results


class TensorflowClassifier:
    def __init__(self, model: tf.keras.models.Model):
        self.model = model
        self.img_size = settings.IMAGE_SIZE

    def preprocess_image(self, input_img) -> np.ndarray:
        """
        Preprocess the input image as needed.
        This step may include resizing, normalizing, or any custom preprocessing required.
        """

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

        return input_img_expanded

    def inference(self, input_img) -> List[Dict]:
        """
        Perform inference on the input image and return TF formatted classification results.
        """

        # Preprocess the image
        img = self.preprocess_image(input_img)

        # Doing inference
        predictions = self.model.predict(img)
        # Post-processing the inference result
        output_list = []

        for prediction in predictions:
            # Getting the index of the class with the highest probability
            predicted_class_index = np.argmax(prediction)

            # Getting the classification score of the predicted class
            classification_score = prediction[predicted_class_index]

            # Adding to the output list
            output_list.append(
                {
                    "class_name": predicted_class_index,
                    "classification_score": classification_score,
                }
            )

        return output_list
