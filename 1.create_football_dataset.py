import os
import glob
import cv2
import shutil
import json
from pprint import pprint

root = "data/football"
splits = ["train", "test"]
output_path = "football_yolo_format/images"
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    shutil.rmtree(output_path.replace("images", "labels"))
os.makedirs(os.path.join(output_path, "train"))
os.makedirs(os.path.join(output_path, "test"))
os.makedirs(os.path.join(output_path.replace("images", "labels"), "train"))
os.makedirs(os.path.join(output_path.replace("images", "labels"), "test"))

for split in splits:
    video_paths = [path.replace(".mp4", "") for path in glob.iglob("{}_{}/*/*.mp4".format(root, split))]
    anno_paths = [path.replace(".json", "") for path in glob.iglob("{}_{}/*/*.json".format(root, split))]
    valid_paths = set(video_paths) & set(anno_paths)
    counter = 0   # global counter
    for path in valid_paths:
        video_path = "{}.mp4".format(path)
        cap = cv2.VideoCapture(video_path)
        anno_path = "{}.json".format(path)
        with open(anno_path, "r") as json_file:
            json_data = json.load(json_file)
        width = json_data["images"][0]["width"]
        height = json_data["images"][0]["height"]
        frame_id = 1    # local counter. Will be reset for each video
        while cap.isOpened():
            print(frame_id)
            flag, frame = cap.read()
            if not flag:
                break
            objects = [anno for anno in json_data["annotations"] if anno["image_id"] == frame_id and int(anno["category_id"]) in [3,4]]
            f = open("{}/{}/{}.txt".format(output_path.replace("images", "labels"), split, counter), mode="w")
            for obj in objects:
                xmin, ymin, ori_width, ori_height = obj["bbox"]
                ##TODO Convert to normalized xcenter, ycenter, width, height
                category_id = obj["category_id"]
                if int(category_id) == 4:
                    class_id = 0
                else:
                    class_id = 1
                # verify that the annotation matches with frame
                # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmin+ori_width), int(ymin+ori_height)), (0, 0, 255), 2)
                xcent = (xmin + ori_width/2)/width
                ycent = (ymin + ori_height/2)/height
                ori_width /= width
                ori_height /= height
                f.write("{} {:06f} {:06f} {:06f} {:06f}\n".format(class_id, xcent, ycent, ori_width, ori_height))
            f.close()
            cv2.imwrite("{}/{}/{}.jpg".format(output_path, split, counter), frame)
            frame_id += 1
            counter += 1

