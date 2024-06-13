import numpy as np

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    result: dict


class MachineLearningDataInput(BaseModel):
    resource: str
    path: str
    httpMethod: str
    isBase64Encoded: bool
    pathParameters: dict
    stageVariables: dict
    headers: dict
    requestContext: dict
    body: str

    def get_prompt(self):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        "Obtenha os cids, Classificação Internacional de Doenças e Problemas Relacionados à Saúde, a partir do texto fornecido em Input"

        ### Input:
        {}

        """.format(
            self.body
        )
        return alpaca_prompt
