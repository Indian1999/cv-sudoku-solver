from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import sudoku_utils_api as sutils
import os
import onnxruntime as ort

app = FastAPI()

MODEL_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_PATH, "models", "model.onnx")

onnx_session = ort.InferenceSession(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Sudoku API is running!"}

@app.post("/read_puzzle_image")
async def read_puzzle_image(file: UploadFile = File(...)):
    image = await file.read()
    image = np.frombuffer(image,np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"message": "Image could not be read."}
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = sutils.resize_and_maintain_aspect_ratio(input_image=image, new_width=1000)

    try:
        cells, M, board_image = sutils.get_valid_cells_from_image(image)
        grid_array = sutils.get_predicted_sudoku_grid_onnx(onnx_session, cells)
        grid_str = sutils.grid_to_puzzle_string(grid_array)
    except Exception as ex:
        print(ex)
        return {"message": "Image could not be read."}

    return {"message": "Image read successfully.", "grid": grid_str}
    