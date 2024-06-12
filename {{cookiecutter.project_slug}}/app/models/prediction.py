import base64

# from io import BytesIO

import numpy as np

# from PIL import Image
from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: dict


class MachineLearningDataInput(BaseModel):
    # A default json file schema
    resource: str
    path: str
    httpMethod: str
    isBase64Encoded: bool
    pathParameters: dict
    stageVariables: dict
    headers: dict
    requestContext: dict
    body: str

    def get_np_array(self):
        """
        This method is used to convert the input data into a numpy array
        """
        return np.array(
            [
                [
                    self.feature1,
                    self.feature2,
                    self.feature3,
                    self.feature4,
                    self.feature5,
                ]
            ]
        )

    def get_image(self):
        """
        This method is used to convert the input data into an image
        """
        try:
            image_bytes = self.body.encode("utf-8")
            image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode="L")
            image = image.resize((28, 28))
            return image
        except KeyError:
            raise ValueError("Invalid input data: 'body' key is missing")
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
