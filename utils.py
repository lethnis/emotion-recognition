import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2


def extract_landmarks(detection_result):
    """Extract x and y coordinates from face landmarks.

    Args:
        detection_result (DetectionResult): detection result from mediapipe model.

    Returns:
        list | None: list with coordinates if detected any face.
    """

    landmarks = detection_result.face_landmarks
    # if there is any face, and number of landmarks is 478
    if len(landmarks) == 1 and len(landmarks[0]) == 478:

        # extract coordinates and store them in temp variable
        temp = []
        for landmark in landmarks[0]:
            x = landmark.x
            y = landmark.y
            temp.extend([x, y])

        return temp
    # if no faces and landmarks found
    return None


def predict_one_image(detection_result, model):
    """Predict emotion on one image.

    Args:
        detection_result (DetectionResult): detection result from mediapipe model.
        model (Estimator): sklearn estimator with predict method.

    Returns:
        list: predicted class.
    """
    landmarks = extract_landmarks(detection_result)
    if landmarks is not None:
        landmarks = np.asarray([landmarks])
        return model.predict(landmarks)
    return ["Unknown"]


def draw_one_image(image, detection_result, model):
    """Draws landmark and predicted class on one image.

    Args:
        image (np.array): image from cv2.imread.
        detection_result (DetectionResult): detection result from mediapipe model.
        model (Estimator): sklearn estimator with predict method.

    Returns:
        image: image with drawn info.
    """

    pred = predict_one_image(detection_result, model)[0]

    h, w, _ = image.shape

    if pred != "Unknown":
        # specify drawing style
        drawing_specs = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
        # change format of keypoints to one mediapipe accepts
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in detection_result.face_landmarks[0]
            ]
        )

        # draw landmarks and text
        mp.solutions.drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_specs,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        cv2.putText(image, pred, (10, h - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    return image
