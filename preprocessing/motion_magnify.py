from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from motion.motion_magnify import motion_magnify


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
Path("output").mkdir(parents=True, exist_ok=True)
input_folder = "../DeepCaps/datasets/casme2/"
output_folder = "casme2_motionM/"
videoExtension = ".mkv"
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
    if np.issubdtype(type(offsetf.values[0]), np.number) and np.issubdtype(
        type(onsetf.values[0]), np.number
    ):
        basePath = (
            input_folder + "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
        )
        onsetPath = basePath + "/reg_img" + str(onsetf.values[0]) + ".jpg"
        onsetImg = cv2.imread(onsetPath)
        offsetPath = basePath + "/reg_img" + str(offsetf.values[0]) + ".jpg"
        offsetImg = cv2.imread(offsetPath)
        if not (isinstance(onsetImg, type(None)) or isinstance(offsetImg, type(None))):
            imgPath = basePath + "/reg_img" + str(onsetf.values[0]) + ".jpg"
            img = cv2.imread(imgPath)
            try:
                w, h, *_ = img.shape
                fourcc = cv2.VideoWriter_fourcc("h", "2", "6", "4")
                basePath = "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
                Path(output_folder + basePath).mkdir(parents=True, exist_ok=True)
                videoOutPath = basePath + "/combinedFrames" + videoExtension
                video = cv2.VideoWriter(
                    output_folder + videoOutPath, fourcc, 20, (h, w)
                )
                img = cv2.imread(imgPath)
                # if not isinstance(img, type(None)):
                resizedImage = cv2.resize(img, (h, w))
                # grayScaledImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
                video.write(resizedImage)
                video.write(resizedImage)
                for j in range(onsetf.values[0], offsetf.values[0]):
                    imgPath = (
                        "sub"
                        + str(subj.values[0]).zfill(2)
                        + "/"
                        + fname.values[0]
                        + "/reg_img"
                        + str(j)
                        + ".jpg"
                    )
                    img = cv2.imread(input_folder + imgPath)
                    resizedImage = cv2.resize(img, (h, w))
                    video.write(resizedImage)
                cv2.destroyAllWindows()
                video.release()
                videoPath = "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
                motion_magnify(
                    video_path=output_folder
                    + videoPath
                    + "/combinedFrames"
                    + videoExtension,
                    output_path=output_folder
                    + videoPath
                    + "/MMFrames"
                    + videoExtension,
                    motion=True,
                    alpha=20,
                    filter_type="butter",
                    low=0.25,
                    high=0.4,
                    lambda_c=16,
                    fps=20,
                )

                MMVideo = cv2.VideoCapture(
                    output_folder
                    + "sub"
                    + str(subj.values[0]).zfill(2)
                    + "/"
                    + fname.values[0]
                    + "/MMFrames"
                    + videoExtension
                )
                success, image = MMVideo.read()
                count = onsetf.values[0]
                while success:
                    basePath = (
                        "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
                    )
                    imgPath = basePath + "/reg_img" + str(count) + ".jpg"
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(
                        output_folder + imgPath,
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, 100],
                    )
                    success, image = MMVideo.read()
                    count += 1

                basePath = "sub" + str(subj.values[0]).zfill(2) + "/" + fname.values[0]
                frame1 = cv2.imread(
                    output_folder
                    + basePath
                    + "/reg_img"
                    + str(onsetf.values[0] + 1)
                    + ".jpg"
                )
                frame2 = cv2.imread(
                    output_folder
                    + basePath
                    + "/reg_img"
                    + str(apexf.values[0] + 1)
                    + ".jpg"
                )

                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                hsv_mask = np.zeros_like(frame1)
                hsv_mask[..., 1] = 255
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv_mask[..., 0] = ang * 180 / np.pi / 2
                hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
                cv2.imwrite(
                    output_folder + basePath + "/OpticalFlow.png", rgb_representation
                )
            except:
                print("Error while loading img: %s" % imgPath)
        else:
            print("Onset:%s" % onsetPath, "Apexset: %s" % offsetPath)
