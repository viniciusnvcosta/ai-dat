from typing import Dict, List, Tuple

import numpy as np

from app.core.config import settings


class YoloDetectorResult:

    def __init__(self):
        self.img_size = settings.IMAGE_SIZE
        pass

    def run_scoring_result(
        self, dict_result_detect, og_img_size: Tuple[int]
    ) -> List[Dict]:
        """
        Build the final detection result object.

        Args:
            dict_result_detect (Dict): The detection result dictionary.
                names (List): The list of class names.
                bboxes (List): The list of bounding boxes.
                classes (List): The list of classes.
                confs (List): The list of confidence scores.
            og_img_size (List[int]): The original image size.


        Returns:
            List[obj (Dict)]: The final detection result object
                class_name (str): The class name.
                detection_score (float): The confidence score.
                bbox (List): The bounding box coordinates.
                    x and y return as the top-left corner of the bounding box + width and height.

        """
        final_score = []
        class_dict = dict_result_detect["names"]

        # Calculate scaling factors for bbox
        scale_x = og_img_size[1] / self.img_size  # width
        scale_y = og_img_size[0] / self.img_size  # height

        for i in range(len(dict_result_detect["bboxes"])):
            # Extract and scale bounding box coordinates
            x_center = dict_result_detect["bboxes"][i][0] * scale_x
            y_center = dict_result_detect["bboxes"][i][1] * scale_y
            width = dict_result_detect["bboxes"][i][2] * scale_x
            height = dict_result_detect["bboxes"][i][3] * scale_y

            # Calculate bbox coordinates
            bbox = [
                int(x_center - (width / 2)),  # x
                int(y_center - (height / 2)),  # y
                int(width),  # width
                int(height),  # height
            ]
            obj = {
                "class_name": class_dict[int(dict_result_detect["classes"][i])],
                "detection_score": dict_result_detect["confs"][i],
                "bbox": bbox,
            }
            final_score.append(obj)

        return final_score


class YoloClassifierResult:

    def __init__(self):
        pass

    def run_scoring_result(self, dict_result_class) -> List[Dict]:
        final_score = []
        for i in range(len(dict_result_class["names"])):
            class_name = dict_result_class["names"][dict_result_class["pred"][i][0]]
            obj = {
                "class_name": class_name,
                "classification_score": max(dict_result_class["softs"][i]),
            }
            final_score.append(obj)
        return final_score
