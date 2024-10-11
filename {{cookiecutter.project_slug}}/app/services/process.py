from typing import Dict, List, Tuple

from ml.model.result_runner import YoloClassifierResult, YoloDetectorResult


class ResultProcessorRunner:

    @classmethod
    def process(cls, runner_class: str, result: Dict, img_shape: Tuple[int]) -> List[Dict]:
        if runner_class == "YoloDetector":
            return YoloDetectorResult().run_scoring_result(result, img_shape)
        elif runner_class == "YoloClassifier":
            return YoloClassifierResult().run_scoring_result(result)
        else:
            return result
