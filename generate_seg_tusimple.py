import json
import numpy as np
import cv2
import os
import argparse

TRAIN_SET = ['trainval.json']
# TRAIN_SET = []
VAL_SET = []
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ['test.json']

def gen_label_for_json(args, image_set):
    H, W = 720, 1280
    SEG_WIDTH = 30
    save_dir = args.savedir

    # Making the ground_truth file.
    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(
        os.path.join(args.root, args.savedir, "list", f"{image_set}_gt.txt"), "w")

    json_path = os.path.join(args.root, args.savedir, f"{image_set}.json")
    with open(json_path) as f:
        for line in f:
            label = json.loads(line)
            # ---------- clean and sort lanes (by X at bottom) -------------

            # _lanes sẽ chứa tất cả các làn đường hợp lệ (x>=0 và >=4 điểm)
            _lanes = []
            # Biến này không cần thiết cho logic mới, nhưng giữ lại tên
            # slope = []

            for i in range(len(label['lanes'])):
                # 1. Lọc bỏ các điểm có x < 0
                l = [(x, y)
                     for x, y in zip(label['lanes'][i], label['h_samples'])
                     if x >= 0]

                if len(l) >= 4:  # Only add lanes with at least 4 points
                    _lanes.append(l)
                # Không cần tính slope nữa

            # Sort only non-empty lanes
            _lanes_sorted = sorted(_lanes, key=lambda lane_coords: lane_coords[-1][0]) if _lanes else []
            
            lanes = _lanes_sorted[:6] + [[]] * (6 - len(_lanes_sorted))
            

            # ---------------------------------------------

            img_path = label['raw_file']
            # Tạo ảnh mask đen hoàn toàn
            seg_img = np.zeros((H, W, 3), dtype=np.uint8)

            list_str_flags = ['0'] * 6

            # --- Vẽ LÊN TẤT CẢ các làn đường hợp lệ đã sắp xếp ---
            # Sử dụng _lanes_sorted để vẽ tất cả các làn đã tìm thấy và lọc
            for i in range(min(6, len(_lanes_sorted))): # Loop qua TẤT CẢ các làn đã sắp xếp
                coords = _lanes_sorted[i] # Lấy tọa độ từ list đã sắp xếp
                
                # Draw lines with grayscale values based on lane index
                for j in range(len(coords) - 1):
                    pt1 = (int(coords[j][0]), int(coords[j][1]))
                    pt2 = (int(coords[j+1][0]), int(coords[j+1][1]))
                    # Kiểm tra điểm hợp lệ (đảm bảo là số nguyên, có thể thêm kiểm tra trong ảnh)
                    if all(p >= 0 for p in pt1) and all(p >= 0 for p in pt2):
                         cv2.line(seg_img, pt1, pt2, (i + 1, i + 1, i + 1), SEG_WIDTH // 2)

            
            for i in range(6):
                 if i < len(_lanes_sorted):
                     list_str_flags[i] = '1' # Đánh dấu làn ở vị trí i là có mặt

            # --- Xây dựng đường dẫn và ghi file ---
            img_name = os.path.basename(img_path)
            
            # Update folder saving logic to match custom version
            seg_path = img_path.split("/")
            seg_path = os.path.join(args.root, args.savedir, seg_path[0])
            os.makedirs(seg_path, exist_ok=True)
            seg_path = os.path.join(seg_path, img_name[:-3] + "png")
            cv2.imwrite(seg_path, seg_img)

            # Update path for list file
            seg_path = "/".join([
                args.savedir, img_name[:-3] + "png"
            ])

            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path

            # Gộp các phần lại thành một dòng cho file list
            list_parts = [img_path, seg_path] + list_str_flags
            list_line = " ".join(list_parts) + "\n"
            list_f.write(list_line)

    list_f.close() # Đóng file list sau khi xử lý xong tất cả ảnh


# Giữ nguyên hai hàm này
def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, json_name)) as infile:
                for line in infile:
                    outfile.write(line)


def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
    ### ~~~ generate_json_file(save_dir, "train_val.json", TRAIN_VAL_SET)
    generate_json_file(save_dir, "trainval.json", TRAIN_VAL_SET)
    generate_json_file(save_dir, "test.json", TEST_SET)
    # 2 files train_val.json, test.json are generated at seg_label/ (This contain all lane-marking in the dataset)

    print("generating train_val set...")
    gen_label_for_json(args, 'trainval')
    print("generating test set...")
    ## ~~~ gen_label_for_json(args, 'test')
    gen_label_for_json(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        required=True,
                        help='The root of the Tusimple dataset')
    parser.add_argument('--savedir',
                        type=str,
                        default='seg_label',
                        help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)