import os
import sys
import pickle
from pathlib import Path

import cv2

import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision

import utils


def main():
    """Program reads files from a directory and detects faces.
    You may specify folder or image in command line the following way:
    'python main.py data/' or 'python main.py data/image.jpg'.
    Default directory is 'assets/' with test images"""

    # load model for predictions
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)

    # get file names
    images, videos = parse_args()

    os.makedirs("output", exist_ok=True)

    save_images(images, model)

    save_videos(videos, model)


def save_images(images_paths, model):
    """Save image with drawn face mesh and predicted emotion."""

    # create options for the model, specify path to the downloaded model
    options = vision.FaceLandmarkerOptions(
        tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    )

    # initialize the model with options
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for image_path in images_paths:
            image = cv2.imread(image_path)
            mp_image = mp.Image(mp.ImageFormat.SRGB, image)
            result = landmarker.detect(mp_image)
            image = utils.draw_one_image(image, result, model)
            cv2.imwrite(f"output/{Path(image_path).name}", image)


def save_videos(videos_paths, model):

    for video_path in videos_paths:

        # create options for the model, specify path to the downloaded model
        options = vision.FaceLandmarkerOptions(
            tasks.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
        )

        # initialize the model with options
        with vision.FaceLandmarker.create_from_options(options) as landmarker:

            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

            # get properties for writer
            w, h, _ = frame.shape
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")

            writer = cv2.VideoWriter(f"output/{os.path.basename(video_path)}", fourcc, fps, (h, w))

            # process frame and read next one
            while ret:
                mp_frame = mp.Image(mp.ImageFormat.SRGB, frame)
                result = landmarker.detect_for_video(mp_frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
                frame = utils.draw_one_image(frame, result, model)
                writer.write(frame)
                ret, frame = cap.read()

            # release memory
            cap.release()
            writer.release()


def parse_args():

    MESSAGE = """Face emotion recognition. Usage:
    Please provide a path to a folder or an image/video file.
    Results will be saved to the 'output' directory."""

    base_dir = "assets"

    if len(sys.argv) == 2:
        base_dir = sys.argv[1]

    assert len(sys.argv) <= 2, f"Too many arguments. \n\n{MESSAGE}"

    assert os.path.exists(base_dir), f"Couldn't find '{base_dir}'. \n\n{MESSAGE}"

    return split_images_videos(base_dir)


def get_paths(base_dir):
    """Get all paths to files in base folder and all subfolders."""

    paths = []

    # if provided path is directory we search inside it
    if os.path.isdir(base_dir):
        # for every file and folder in base dir
        for i in os.listdir(base_dir):
            # update path so it will be full
            new_path = os.path.join(base_dir, i)
            # check if new path is a file or a folder
            if os.path.isdir(new_path):
                paths.extend(get_paths(new_path))
            else:
                paths.append(new_path)

    # Else path is a file, just add it to the resulting list
    else:
        paths.append(base_dir)

    return paths


def split_images_videos(base_dir):
    """Take all paths and extract only image and video formats."""

    paths = get_paths(base_dir)

    images, videos = [], []

    for path in paths:
        if any([path.endswith(i) for i in [".png", ".jpg", "jpeg"]]):
            images.append(path)
        elif any([path.endswith(i) for i in [".mp4", ".avi"]]):
            videos.append(path)

    return images, videos


if __name__ == "__main__":
    main()
