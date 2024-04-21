import pickle

import cv2

import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision

import utils


def main():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    with open("model.pickle", "rb") as f:
        model = pickle.load(f)

    # create options for the model, specify path to the downloaded model
    options = vision.FaceLandmarkerOptions(
        tasks.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
    )

    # initialize the model with options
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Press 'q' to quit.")
        while ret:
            mp_frame = mp.Image(mp.ImageFormat.SRGB, frame)
            result = landmarker.detect_for_video(mp_frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            frame = utils.draw_one_image(frame, result, model)
            cv2.imshow("webcam", frame)
            ret, frame = cap.read()
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
