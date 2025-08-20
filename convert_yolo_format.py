import os
import cv2
import json

# Đường dẫn dữ liệu gốc (video)
path = "data/football_train"  # hoặc "data/football_test"
# Đường dẫn output cho YOLO dataset
output = "yolo_football"
sub_dirs = ["train", "valid", "test"]

# Nếu chưa có thư mục output thì tạo
if not os.path.exists(output):
    os.makedirs(output)

# Tạo cấu trúc con: train/valid/test với images và labels
for sub_dir in sub_dirs:
    dir_path = os.path.join(output, sub_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    images_path = os.path.join(dir_path, "images")
    labels_path = os.path.join(dir_path, "labels")
    
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

# Duyệt qua tất cả video trong thư mục
for dir_name in os.listdir(path):
    dir_path = os.path.join(path, dir_name)
    
    if not os.path.isdir(dir_path):
        continue
        
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(dir_path, file_name)
            print("Đang xử lý:", file_path)
            
            # Tìm file JSON annotation tương ứng
            json_path = file_path.replace(".mp4", ".json")
            annotations = []
            images_info = []
            
            if os.path.exists(json_path):
                print(f"Đọc annotations từ: {json_path}")
                with open(json_path, "r") as f:
                    data = json.load(f)
                    annotations = data.get("annotations", [])
                    images_info = data.get("images", [])
                    print(f"Số annotations: {len(annotations)}")
                    print(f"Số images: {len(images_info)}")
                    
                    # Debug: xem structure của data
                    if len(annotations) > 0:
                        print("Sample annotation:", annotations[0])
                    if len(images_info) > 0:
                        print("Sample image info:", images_info[0])
            else:
                print(f"Không tìm thấy file JSON: {json_path}")
            
            # Xác định mode dựa trên tên đường dẫn
            if "train" in path:
                mode = "train"
            elif "test" in path:
                mode = "test"  
            else:
                mode = "valid"
            
            # Mở video bằng OpenCV
            cap = cv2.VideoCapture(file_path)
            counter = 1
            
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                
                # In ra kích thước frame để kiểm tra  
                print(f"{file_name} - Frame {counter}: {frame.shape}")
                frame_height, frame_width = frame.shape[:2]
                
                # Tìm image_id tương ứng với frame hiện tại
                current_image_id = None
                if counter <= len(images_info):
                    # Tìm image có file_name tương ứng với frame counter
                    for img_info in images_info:
                        if img_info.get("id") == counter or img_info.get("frame_id") == counter:
                            current_image_id = img_info["id"]
                            break
                
                # Nếu không tìm thấy mapping, dùng counter làm image_id
                if current_image_id is None:
                    current_image_id = counter
                
                print(f"Frame {counter}, Image ID: {current_image_id}")
                
                # Lưu frame thành ảnh JPG
                output_image_path = os.path.join(output, mode, "images", 
                                               f"{file_name.replace('.mp4', '')}_{counter:04d}.jpg")
                cv2.imwrite(output_image_path, frame)
                
                # Tạo file label với annotations tương ứng
                output_label_path = os.path.join(output, mode, "labels", 
                                               f"{file_name.replace('.mp4', '')}_{counter:04d}.txt")
                
                # Tìm annotations cho image_id hiện tại
                current_annotations = [anno for anno in annotations if anno.get("image_id") == current_image_id]
                print(f"Found {len(current_annotations)} annotations for frame {counter}")
                
                with open(output_label_path, 'w') as f:
                    for anno in current_annotations:
                        # Kiểm tra xem annotation có đầy đủ thông tin không
                        if "bbox" not in anno or "category_id" not in anno:
                            continue
                            
                        # Chuyển đổi bbox từ format COCO sang YOLO
                        bbox = anno["bbox"]  # [x, y, width, height] trong COCO
                        x, y, w, h = bbox
                        
                        # Chuyển sang center format và normalize
                        center_x = (x + w/2) / frame_width
                        center_y = (y + h/2) / frame_height
                        norm_width = w / frame_width
                        norm_height = h / frame_height
                        
                        # Class ID (category_id - có thể cần trừ 1 tùy vào dataset)
                        class_id = anno["category_id"]
                        
                        # Đảm bảo giá trị nằm trong khoảng [0, 1]
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))
                        
                        # Ghi vào file theo format YOLO
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                counter += 1
            
            cap.release()
            print(f"Đã xử lý xong {file_name} với {counter-1} frames")

print("Hoàn thành trích xuất dataset!")