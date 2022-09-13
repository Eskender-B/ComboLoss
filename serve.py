from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, status, Request
from typing import List
from pydantic import BaseModel, BaseConfig, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from PIL import Image
from io import BytesIO
from main.inference import FacialBeautyPredictor

# api defnition
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
fbp = FacialBeautyPredictor(pretrained_model_path='/data/ComboNet_SCUTFBP5500.pth')

# Input data model
class Inputs(BaseModel):
    image_url: str


@app.get('/')
async def hello_world() -> str:
    """ Standard get call to check if service is active
    Returns:
        str: message
    """
    return 'Hello ComboLoss'


@app.post('/api/v1/comboloss_inference/')
async def score(input_data: Inputs) -> object:
    """
        It takes a url of a face image and returns a score.
    """
    
    try:
        url = input_data.dict()['image_url']
        resp = requests.get(url)
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"message": "Failed to get input image"})

        img_bytes = BytesIO(resp.content)

        res = fbp.infer(img_bytes)

        return JSONResponse(status_code=200, content=res)
        
    except Exception as exc:
        return JSONResponse(status_code=500, content={"message": "Server Error {}".format(str(exc))})


if __name__ == '__main__':
    app.run()
