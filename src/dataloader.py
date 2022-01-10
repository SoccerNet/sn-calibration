"""
DataLoader used to train the segmentation network used for the prediction of extremities.
"""

import json
import os
import time
import zipfile
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import torch
import torchvision
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY, INVERSE_FRAME_CLASS_DICTIONARY
from SoccerNet.utils import getListGames
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.soccerpitch import SoccerPitch


class SNV3Dataset(Dataset):

    def __init__(self, path, split="all", resolution=(1920, 1080), preload_images=False, tiny=None,
                 zipped_images=False):

        # Path for the SoccerNet-v3 dataset
        # containing the images and labels
        self.path = path

        # Get the list of the selected subset of games
        self.list_games = getListGames(split, task="frames")
        if tiny is not None:
            self.list_games = self.list_games[:tiny]

        # Resolution of the images to load (width, height)
        self.resolution = resolution
        self.resize = torchvision.transforms.Resize((resolution[1], resolution[0]), antialias=True)
        self.preload_images = preload_images
        self.zipped_images = zipped_images

        # Variable to store the metadata
        print("Reading the annotation files")
        self.metadata = list()
        for game in tqdm(self.list_games):
            self.metadata.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))

        # Variables to store the preloaded images and annotations
        # Each element in the list is a list of images and annotations linked to an action
        self.data = list()
        for annotations in tqdm(self.metadata):

            # Retrieve each action in the game
            for action_name in annotations["GameMetadata"]["list_actions"]:

                # concatenate the replays of each action with itself
                img_list = [action_name] + annotations["actions"][action_name]["linked_replays"]
                self.data.append(list())
                IDs_list = list()

                zipfilepath = os.path.join(self.path, annotations["GameMetadata"]["UrlLocal"], 'Frames-v3.zip')
                if self.zipped_images:
                    zippedFrames = zipfile.ZipFile(zipfilepath, 'r')

                # For each image extract the images and annotations
                for i, img in enumerate(img_list):

                    # Variable to save the annotation
                    data_tmp = dict()
                    data_tmp["match_id"] = annotations["GameMetadata"]["UrlLocal"]
                    data_tmp["image"] = None

                    # Only the first frame is an action, the rest are replays
                    img_type = "actions"
                    if i > 0:
                        img_type = "replays"

                    filepath = os.path.join(self.path, annotations["GameMetadata"]["UrlLocal"], "v3_frames", img)
                    data_tmp["zipfilepath"] = zipfilepath
                    data_tmp["imagefilepath"] = img
                    data_tmp["filepath"] = filepath

                    data_tmp["bboxes"], ID_tmp = self.format_bboxes(annotations[img_type][img]["bboxes"],
                                                                    annotations[img_type][img]["imageMetadata"])

                    data_tmp["lines"] = self.format_lines(annotations[img_type][img]["lines"],
                                                          annotations[img_type][img]["imageMetadata"])

                    n_lines = len(data_tmp["lines"])
                    data_tmp["links"] = None

                    if n_lines >= 4:
                        self.data[-1].append(data_tmp)

                    IDs_list.append(ID_tmp)

    def format_bboxes(self, bboxes, image_metadata):

        # Bounding boxes in x_top, y_top, width, height, cls_idx, num_idx
        data = list()

        IDs = list()

        for i, bbox in enumerate(bboxes):

            if bbox["class"] is not None:

                tmp_data = torch.zeros((4 + 1 + 1,), dtype=torch.float) - 1
                tmp_data[0] = bbox["points"]["x1"] / image_metadata["width"]
                tmp_data[1] = bbox["points"]["y1"] / image_metadata["height"]
                tmp_data[2] = abs(bbox["points"]["x2"] - bbox["points"]["x1"]) / image_metadata["width"]
                tmp_data[3] = abs(bbox["points"]["y2"] - bbox["points"]["y1"]) / image_metadata["height"]
                tmp_data[4] = float(FRAME_CLASS_DICTIONARY[bbox["class"]])
                if bbox["ID"] is not None:
                    if bbox["ID"].isnumeric():
                        tmp_data[5] = float(bbox["ID"])
                IDs.append([bbox["ID"], FRAME_CLASS_DICTIONARY[bbox["class"]]])
                data.append(tmp_data)

        data = torch.stack(data)
        return data, IDs

    @staticmethod
    def format_lines(lines, image_metadata):
        # Each element is a list with list of points, cls_idx
        data = list()

        for line in lines:

            if line["class"] is not None:
                points = torch.FloatTensor(line["points"])
                points[::2] = points[::2] / image_metadata["width"]
                points[1::2] = points[1::2] / image_metadata["height"]
                data.append([points, torch.FloatTensor([FRAME_CLASS_DICTIONARY[line["class"]]])])
        return data

    def format_links(self, IDs_list):

        # Links are stored as (index of the current image,
        # index of the bounding box in the first image,
        # index of the second image,
        # index of the bounding box in the second image)

        for i, IDs_1 in enumerate(IDs_list):

            list_of_links = list()

            for j, IDs_2 in enumerate(IDs_list):

                if i == j:
                    continue

                for k, ID_1 in enumerate(IDs_1):

                    for l, ID_2 in enumerate(IDs_2):
                        if ID_1[1] == ID_2[1]:
                            if ID_1[0] is not None and ID_2[0] is not None:
                                if ID_1[0] == ID_2[0]:
                                    list_of_links.append([i, k, j, l])
                                    continue

            self.data[-1][i]["links"] = torch.FloatTensor(list_of_links)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):

        return len(self.data)


class SoccerNetDataset(Dataset):
    def __init__(self,
                 datasetpath,
                 split="test",
                 width=640,
                 height=360,
                 mean="../resources/mean.npy",
                 std="../resources/std.npy"):
        self.mean = np.load(mean)
        self.std = np.load(std)
        self.width = width
        self.height = height
        dataset = SNV3Dataset(datasetpath, split=split)
        self.data = []
        self.n_samples = 0
        for item in dataset:
            self.n_samples += len(item)
            for element in item:

                groundtruth_lines = {}
                img = element['filepath']

                for line in element["lines"]:
                    class_name = INVERSE_FRAME_CLASS_DICTIONARY[line[1].item()]
                    if class_name in SoccerPitch.lines_classes:
                        points_list = []
                        for i, point in enumerate(line[0]):
                            if i % 2 == 0:
                                x = point.item()
                            else:
                                y = point.item()
                                points_list.append((x, y))

                        groundtruth_lines[class_name] = points_list
                self.data.append({
                    "image_path": img,
                    "annotations": groundtruth_lines,
                    "match_id": element["match_id"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img = cv.imread(item["image_path"])
        img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_LINEAR)

        mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        img = np.asarray(img, np.float32) / 255.
        img -= self.mean
        img /= self.std
        img = img.transpose((2, 0, 1))
        for class_number, class_ in enumerate(SoccerPitch.lines_classes):
            if class_ in item["annotations"].keys():
                key = class_
                line = item["annotations"][key]

                prev_point = line[0]
                for i in range(1, len(line)):
                    next_point = line[i]
                    cv.line(mask,
                            (int(prev_point[0] * mask.shape[1]), int(prev_point[1] * mask.shape[0])),
                            (int(next_point[0] * mask.shape[1]), int(next_point[1] * mask.shape[0])),
                            class_number + 1,
                            2)
                    prev_point = next_point
        return img, mask


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description='dataloader')

    parser.add_argument('--SoccerNet_path', default= "/mnt/ahl03/data/football/SoccerNet/data/", type=str, help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    parser.add_argument('--split', required=False, type=str, default="all", help='Select the split of data')
    parser.add_argument('--num_workers', required=False, type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--resolution_width', required=False, type=int, default=1920,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=1080,
                        help='height resolution of the images')
    parser.add_argument('--preload_images', action='store_true',
                        help="Preload the images when constructing the dataset")
    parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    args = parser.parse_args()

    start_time = time.time()
    soccernet = SoccerNetDataset(args.SoccerNet_path, split=args.split)
    with tqdm(enumerate(soccernet), total=len(soccernet), ncols=160) as t:
        for i, data in t:
            img = soccernet[i][0].astype(np.uint8).transpose((1, 2, 0))
            print(img.shape)
            print(img.dtype)
            cv.imshow("Normalized image", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(data[1].shape)
            cv.imshow("Mask", soccernet[i][1].astype(np.uint8))
            cv.waitKey(0)
            cv.destroyAllWindows()
            continue
    end_time = time.time()
    print(end_time - start_time)
