from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.python.solutions.drawing_utils import DrawingSpec


def loading_casme_labels(objective_flag):
    label_file = pd.read_excel("CASME2-coding-20190701.xlsx", sheet_name="Sheet1")
    # remove class 6, 7
    if objective_flag:
        label_file = label_file.loc[label_file["Estimated Emotion"] != "others"]
    subject = label_file[["Subject"]]
    filename = label_file[["Filename"]]
    label = label_file[["Estimated Emotion"]]
    onset = label_file[["OnsetFrame"]]
    apex = label_file[["ApexFrame"]]
    offset = label_file[["OffsetFrame"]]
    return subject, filename, label, onset, apex, offset


subject, filename, label, onset, apex, offset = loading_casme_labels(False)


def get_ROI(part):
    pts = []
    for index in part:
        shape = image.shape
        x = results.multi_face_landmarks[0].landmark[index].x
        y = results.multi_face_landmarks[0].landmark[index].y
        x, y = int(x * shape[1]), int(y * shape[0])
        pt = [x, y]
        pts.append(pt)
    return np.array(pts)


Path("casme2_landmarks").mkdir(parents=True, exist_ok=True)
img_folder = "E:/python/casme2_loaders/output/videos/"

mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

for (
    (idx1, onsetf),
    (idx2, apexf),
    (idx3, subj),
    (idx4, fname),
    (idx5, label),
    (idx6, offsetf),
) in zip(
    onset.iterrows(),
    apex.iterrows(),
    subject.iterrows(),
    filename.iterrows(),
    label.iterrows(),
    offset.iterrows(),
):
    if np.issubdtype(type(apexf.values[0]), np.number) and np.issubdtype(
        type(onsetf.values[0]), np.number
    ):
        onsetPath = (
            "sub"
            + str(subj.values[0]).zfill(2)
            + "/"
            + fname.values[0]
            + "/reg_img"
            + str(onsetf.values[0])
            + ".jpg"
        )
        onsetImg = cv2.imread(img_folder + onsetPath)
        offsetPath = (
            "sub"
            + str(subj.values[0]).zfill(2)
            + "/"
            + fname.values[0]
            + "/reg_img"
            + str(offsetf.values[0])
            + ".jpg"
        )
        offsetImg = cv2.imread(img_folder + offsetPath)
        if not (isinstance(onsetImg, type(None)) or isinstance(offsetImg, type(None))):
            for i in range(onsetf.values[0] + 1, offsetf.values[0] + 2):
                basePath = "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
                constructedpath = basePath + "/reg_img" + str(i)
                image = cv2.imread(img_folder + constructedpath + ".jpg")
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                ) as face_mesh:
                    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not results.multi_face_landmarks:
                        continue
                    annotated_image = image

                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=2
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=2
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=2
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=2
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_LIPS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=2
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=DrawingSpec(
                                color=(180, 180, 180), thickness=1, circle_radius=1
                            ),
                        )
                        Path("casme2_landmarks/" + basePath).mkdir(
                            parents=True, exist_ok=True
                        )
                        cpath = (
                            "casme2_landmarks/"
                            + basePath
                            + "/reg_img"
                            + str(i)
                            + ".jpg"
                        )
                        annotated_image = np.float32(annotated_image)
                        cv2.imwrite(filename=cpath, img=annotated_image)
        else:
            print("Onset:%s" % onsetPath, "Offset: %s" % offsetPath)
