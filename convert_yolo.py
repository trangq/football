import os
import cv2
import json

# Đường dẫn và cấu hình
path = "data/football_train"
output = "yolo_football"
sub_dirs = ["train", "valid", "test"]

# Tạo thư mục output nếu chưa tồn tại
if not os.path.isdir(output):
    os.makedirs(output)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(output, sub_dir))
        os.makedirs(os.path.join(output, sub_dir, "images"))
        os.makedirs(os.path.join(output, sub_dir, "labels"))

# Duyệt qua các thư mục con
for sub_dir in sub_dirs:
    dir_path = os.path.join(output, sub_dir)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, "images"))
        os.makedirs(os.path.join(dir_path, "labels"))

# Xử lý các file video
for dir_name in os.listdir(path):
    dir_path = os.path.join(path, dir_name)
    for file_name in os.listdir(dir_path):
        if ".mp4" in file_name:
            file_path = os.path.join(dir_path, file_name)
            cap = cv2.VideoCapture(file_path)
            counter = 1
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                # Xác định mode dựa trên đường dẫn
                if "train" in path:
                    mode = "train"
                else:
                    mode = "valid"
                # Lưu frame thành ảnh
                cv2.imwrite("{}/{}/images/{}_{}.jpg".format(
                    output, mode, file_name.replace(".mp4", ""), counter), frame)
                counter += 1
                # Đọc file annotation JSON
                with open(file_path.replace(".mp4", ".json"), "r") as f:
                    data = json.load(f)
                    annotations = data["annotations"]
                    num_frames = len(data["images"])
                    # Lọc annotations cho frame hiện tại
                    for frame_id in range(num_frames):
                        current_annotations = [anno for anno in annotations
                                             if anno["image_id"] == frame_id + 1 and anno["category_id"] == 4]
                        
                        # Duyệt qua các bounding box (bổ sung từ ảnh 2)
                        for bbox_data in current_annotations:
                            # Lấy tọa độ bbox từ dữ liệu JSON
                            x, y, w, h = bbox_data["bbox"]
                            # Tính toán tọa độ hình chữ nhật (bổ sung từ ảnh 2)
                            xmin = int(x - w / 2)
                            xmax = int(x + w / 2)
                            ymin = int(y - h / 2)
                            ymax = int(y + h / 2)
                            # Vẽ hình chữ nhật trên frame (bổ sung từ ảnh 2)
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

                # Lưu ảnh test (có vẻ như bạn muốn lưu ảnh đã vẽ box lên)
                cv2.imwrite(filename="test.jpg", img=frame)
            # Giải phóng capture
            cap.release()
# Thoát chương trình
exit(0)