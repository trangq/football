import os
import cv2
import json
import glob

output_crop = "football_crops"
os.makedirs(os.path.join(output_crop, "train"), exist_ok=True)
os.makedirs(os.path.join(output_crop, "test"), exist_ok=True)

root = "data/football"
splits = ["train", "test"]

for split in splits:
    video_paths = [path.replace(".mp4", "") for path in glob.iglob(f"{root}_{split}/*/*.mp4")]
    anno_paths = [path.replace(".json", "") for path in glob.iglob(f"{root}_{split}/*/*.json")]
    valid_paths = set(video_paths) & set(anno_paths)
    counter = 0
    for path in valid_paths:
        video_path = f"{path}.mp4"
        anno_path = f"{path}.json"
        with open(anno_path, "r") as json_file:
            json_data = json.load(json_file)
        cap = cv2.VideoCapture(video_path)
        frame_id = 1
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break
            objects = [anno for anno in json_data["annotations"] if anno["image_id"] == frame_id]
            for obj_idx, obj in enumerate(objects):
                xmin, ymin, w, h = obj["bbox"]
                x1, y1 = int(xmin), int(ymin)
                x2, y2 = int(xmin + w), int(ymin + h)
                crop_img = frame[y1:y2, x1:x2]
                # Gán nhãn số áo
                shirt_number = obj.get("shirt_number", None)
                if shirt_number is None:
                    crop_class = 0
                elif 1 <= shirt_number <= 10:
                    crop_class = shirt_number
                elif shirt_number >= 11:
                    crop_class = 11
                else:
                    crop_class = 0
                crop_name = f"{split}_{counter}_{obj_idx}_cls{crop_class}.jpg"
                crop_path = os.path.join(output_crop, split, crop_name)
                cv2.imwrite(crop_path, crop_img)
            frame_id += 1
            counter += 1