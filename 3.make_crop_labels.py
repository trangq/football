import os

output_crop = "football_crops"
label_dir = "football_crops_labels"
os.makedirs(label_dir, exist_ok=True)

splits = ["train", "test"]

for split in splits:
    label_lines = []
    crop_dir = os.path.join(output_crop, split)
    for img_name in os.listdir(crop_dir):
        if "_cls" in img_name:
            crop_class = img_name.split("_cls")[-1].split(".")[0]
            label_lines.append(f"{img_name} {crop_class}")
    with open(os.path.join(label_dir, f"{split}.txt"), "w") as f:
        for line in label_lines:
            f.write(line + "\n")