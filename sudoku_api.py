from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import sudoku_utils as sutils
import tensorflow as tf
import os

app = FastAPI()

# Tensorflow was unable to load the original 'model_15_epochs_font_mnist.keras'
# so I trained a new one
MODEL_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_PATH, "models", "model_15_epochs_both.keras")

@app.get("/")
async def root():
    return {"message": "Sudoku API is running!"}

@app.post("/read_puzzle_image")
async def read_puzzle_image(file: UploadFile = File(...)):
    """
    Asynchronously read and process an uploaded Sudoku puzzle image, run a trained TensorFlow model to
    predict the digits in each cell, and return a JSON-serializable result.

    Parameters
    ----------
    file : UploadFile
        A Starlette/FastAPI UploadFile (provided via File(...)) containing the image bytes to be
        processed. The function reads the file contents asynchronously.
    Returns
    -------
    dict
        A dictionary suitable for returning from a FastAPI endpoint. On success, the dictionary has the
        form:
            {"message": "Image read successfully.", "grid": `str`}
        where `str` is a single string containing the predicted digits for the Sudoku grid in (e.g.,"530070000600195000098000060800060003400803001700020006060000280000419005000080079")
        row-major order (e.g., 81 characters for a 9x9 grid). On failure to decode the image, the
        function returns:
            {"message": "Image could not be read."}
    """
    image = await file.read()
    image = np.frombuffer(image,np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = sutils.resize_and_maintain_aspect_ratio(input_image=image, new_width=1000)

    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    cells, M, board_image = sutils.get_valid_cells_from_image(image)
    grid_array = sutils.get_predicted_sudoku_grid(loaded_model, cells)
    grid_str = sutils.grid_to_puzzle_string(grid_array)

    if image is None:
        return {"message": "Image could not be read."}
    return {"message": "Image read successfully.", "grid": grid_str}
    