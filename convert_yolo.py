import os
import cv2
import json

# Đường dẫn dữ liệu gốc (video)
path = "data/football_train"  # hoặc "data/football_test"
# Đường dẫn output cho YOLO dataset
output = "yolo_football"
sub_dirs = ["train", "valid", "test"]

# Biến để kiểm soát việc lưu ảnh test
saved_test_image = False

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
            
            # PHẦN 1: Đọc JSON annotation tương ứng
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
                
                # PHẦN 1: Tìm image_id tương ứng với frame hiện tại
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
                
                # PHẦN 1: Lưu frame thành ảnh JPG
                output_image_path = os.path.join(output, mode, "images", 
                                               f"{file_name.replace('.mp4', '')}_{counter:04d}.jpg")
                cv2.imwrite(output_image_path, frame)
                
                # PHẦN 1: Tạo file label với annotations tương ứng
                output_label_path = os.path.join(output, mode, "labels", 
                                               f"{file_name.replace('.mp4', '')}_{counter:04d}.txt")
                
                # PHẦN 1: Tìm annotations cho image_id hiện tại
                current_annotations = [anno for anno in annotations if anno.get("image_id") == current_image_id]
                print(f"Found {len(current_annotations)} annotations for frame {counter}")
                
                # Tạo bản copy của frame để vẽ box
                frame_with_boxes = frame.copy()
                
                with open(output_label_path, 'w') as f:
                    for anno in current_annotations:
                        # Kiểm tra xem annotation có đầy đủ thông tin không
                        if "bbox" not in anno or "category_id" not in anno:
                            continue
                            
                        # PHẦN 1: Chuyển đổi bbox từ format COCO sang YOLO
                        bbox = anno["bbox"]  # [x, y, width, height] trong COCO
                        x, y, w, h = bbox
                        
                        # Chuyển sang center format và normalize
                        center_x = (x + w/2) / frame_width
                        center_y = (y + h/2) / frame_height
                        norm_width = w / frame_width
                        norm_height = h / frame_height
                        
                        class_id = anno["category_id"]
                        
                        # Đảm bảo giá trị nằm trong khoảng [0, 1]
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))
                        
                        # Ghi vào file theo format YOLO
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                        
                        # PHẦN 2: Vẽ bounding box lên frame 
                        xmin = int(x - w / 2)  
                        xmax = int(x + w / 2)
                        ymin = int(y - h / 2)
                        ymax = int(y + h / 2)
                        
                        # Vẽ hình chữ nhật trên frame copy
                        cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                        
                        # Thêm label text
                        label_text = f"Class_{class_id}"
                        cv2.putText(frame_with_boxes, label_text, (xmin, ymin-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # PHẦN 2: Lưu ảnh test (chỉ lưu 1 ảnh đầu tiên có annotations)
                if len(current_annotations) > 0 and not saved_test_image:
                    cv2.imwrite("test.jpg", frame_with_boxes)
                    print(f"✅ Đã lưu test.jpg với {len(current_annotations)} bounding boxes")
                    saved_test_image = True
                
                counter += 1
            
            # Giải phóng capture
            cap.release()
            print(f"Đã xử lý xong {file_name} với {counter-1} frames")

print("Hoàn thành trích xuất dataset!")
