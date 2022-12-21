import cv2
import numpy as np
import pandas as pd


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
    # print(label)
    return subject, filename, label, onset, apex, offset


subject, filename, label, onset, apex, offset = loading_casme_labels(False)

encodedLabels = pd.DataFrame(
    {
        "disgust": 0,
        "fear": 1,
        "happiness": 2,
        "others": 3,
        "repression": 4,
        "sadness": 5,
        "surprise": 6,
    },
    index=[0],
)
# F=pd.DataFrame({"Subject": ["..."], "Clip": ["..."], "Label": ["..."], "ApexFrame": ["..."], "DynamicFeatures": ["..."]})
F = pd.DataFrame(
    {"Subject": ["..."], "Clip": ["..."], "Label": ["..."], "ApexFrame": ["..."]}
)
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
        onsetImg = cv2.imread("../DeepCaps-main/dataset_folder/casme2/" + onsetPath)
        offsetPath = (
            "sub"
            + str(subj.values[0]).zfill(2)
            + "/"
            + fname.values[0]
            + "/reg_img"
            + str(offsetf.values[0])
            + ".jpg"
        )
        offsetImg = cv2.imread("../DeepCaps-main/dataset_folder/casme2/" + offsetPath)
        if not (isinstance(onsetImg, type(None)) or isinstance(offsetImg, type(None))):
            subjPath = "sub" + str(subj.values[0]).zfill(2)
            # F = pd.concat([F, pd.DataFrame({"Subject": subjPath, "Clip": fname.values[0], "Label": encodedLabels[label].values[0], "ApexFrame": apexf.values[0] + 1, "DynamicFeatures": onsetf.values[0]})], ignore_index = True)
            F = pd.concat(
                [
                    F,
                    pd.DataFrame(
                        {
                            "Subject": subjPath,
                            "Clip": fname.values[0],
                            "Label": encodedLabels[label].values[0],
                            "ApexFrame": apexf.values[0] + 1,
                        }
                    ),
                ],
                ignore_index=True,
            )
# Remove the placeholder row.
finalDf = F.drop([0])
finalDf.to_csv("casme2 of ahhhh.csv", index=True, index_label="Id")
print("Conversion is now complete! To the stars and beyond!")
