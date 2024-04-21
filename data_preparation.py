from tqdm import tqdm
import pickle
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision

from utils import extract_landmarks


def main():
    """Program takes images from 'emotions' folder, extracts face landmarks
    and saves class names and landmarks to the 'data.pickle' file.
    """

    # dir with classes
    base_dir = Path("emotions")
    # get paths to all images
    images = list(base_dir.glob("*/*"))
    # create dict to store classes and landmarks
    data = {"class": [], "landmarks": []}

    # create options for the model, specify path to the downloaded model
    options = vision.FaceLandmarkerOptions(
        tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    )

    # initialize the model with options
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for image in tqdm(images):
            # read current image as mp.Image type
            mp_image = mp.Image.create_from_file(str(image))
            # detect face and its landmarks
            result = landmarker.detect(mp_image)
            # extract landmarks as vector with shape (956,)
            landmarks = extract_landmarks(result)

            if landmarks is not None:
                # add class name and all landmarks coordinates to the data
                data["class"].append(image.parent.name)
                data["landmarks"].append(landmarks)

    # change data type to the np.array
    data["class"] = np.asarray(data["class"])
    data["landmarks"] = np.asarray(data["landmarks"])

    # save data as pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
