import os
import cv2
import cfg
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import RandomRotation


class CASME2:
    def __init__(
        self,
        data_path,
        batch_size,
        shuffle,
        type,
        num_workers=4,
        rotation_degrees=30,
        translate=(0, 0.2),
        scale=(0.95, 1.2),
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.type = type
        self.img_size = 28
        self.num_class = 7

    def __call__(self):
        df = pd.read_csv("./datasets/casme2_da.csv")
        test_subject = "10"
        test_set = df[df['Subject'] == f"sub{test_subject}"]
        train_set = pd.concat([df,test_set]).drop_duplicates(keep=False)
        # train_set, test_set = train_test_split(df, test_size=0.15)
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = CASME2Dataset(
            csv=train_set,
            img_folder=self.data_path,
            transform=train_transform,
            type=self.type,
        )
        test_dataset = CASME2Dataset(
            csv=test_set,
            img_folder=self.data_path,
            transform=test_transform,
            type=self.type,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return train_loader, test_loader, self.img_size, self.num_class


class CASME2Dataset(Dataset):
    def __init__(self, csv, img_folder, transform, type):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.type = type
        self.image_names = self.csv[:][
            ["Subject", "Clip", "ApexFrame"]
        ]
        self.labels = np.array(
            self.csv.drop(
                ["Id", "Subject", "Clip", "ApexFrame"],
                axis=1
            )
        )
        self.isOrig = True

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        idx = self.image_names.iloc[index]

        dir_path = "casme2_landmarks/" + str(idx["Subject"]) + "/" + idx["Clip"] + "/"
        apex_path = dir_path + str(idx["ApexFrame"])
        apex_image = cv2.imread(self.img_folder + apex_path, 0)
        apex_image = self.transform(apex_image)
        if(self.type == 'optic'):
            dir_path = "casme2_landmarks/" + str(idx["Subject"]) + "/" + idx["Clip"] + "/"
            apex_path =  dir_path + "OpticalFlow"
            
        optic_image = cv2.imread(self.img_folder + apex_path + ".png", 0)
        optic_image = self.transform(optic_image)
        labels = self.labels[index]
        labels = labels.squeeze()

        sample = (
            apex_image,
            optic_image,
            labels,
        )
        

        return sample
