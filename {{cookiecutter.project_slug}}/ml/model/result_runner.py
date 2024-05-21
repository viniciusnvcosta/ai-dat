import numpy as np
import math

class YoloDetectorResult():

    def __init__(self):
        pass

    def run_scoring_result(self, dict_result_detect):
        final_score = []
        class_dict = dict_result_detect["names"]
        for i in range(len(dict_result_detect['bboxes'])):
            bbox = [int(dict_result_detect["bboxes"][i][0] - (dict_result_detect["bboxes"][i][2]/2)),int(dict_result_detect["bboxes"][i][1] - (dict_result_detect["bboxes"][i][3]/2)),int(dict_result_detect["bboxes"][i][2]),int(dict_result_detect["bboxes"][i][3])]
            obj = {"bbox": bbox,
                   "class_name": class_dict[int(dict_result_detect["classes"][i])],
                   "detection_score": dict_result_detect["confs"][i] }
            final_score.append(obj)

        return final_score

class YoloClassifierResult():

    def __init__(self):
        pass

    def run_scoring_result(self, dict_result_class):
        final_score = []
        for i in range(len(dict_result_class["names"])):
            class_name = dict_result_class["names"][dict_result_class["pred"][i][0]]
            obj = {
                "class_name": class_name,
                "classification_score": max(dict_result_class["softs"][i]),
            }
            final_score.append(obj)
        return final_score