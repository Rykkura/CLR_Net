import numpy as np
import cv2
import os
import json
import argparse
from tqdm import tqdm

def genmaps(H, W, plist):
    line = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H, 10):
        cv2.line(line, (0, i), (W, i), (255), 1)
    
    for i in range(len(plist)-1):
        cv2.line(mask, (int(plist[i][0]), int(plist[i][1])), 
                        (int(plist[i+1][0]), int(plist[i+1][1]) ), (255), 1)
    
    return mask.astype(int), line.astype(int)

def create_anno(args):
    root_folder = f'{args.root}'
    height_separation = 10 # <-- Use 10 as default.
    json_f = open(f'{root_folder}/{args.out_file}.json', 'w')

    # Loop through all labelme json files and convert to TuSimple format.
    for path in tqdm(os.listdir(root_folder)):
        if not path.endswith('json') or path.startswith(f'{args.out_file}'): continue
        with open(os.path.join(root_folder, path), 'r') as f:
            data = json.load(f)
        H, W = data['imageHeight'], data['imageWidth']
        filePath = data['imagePath']
        lanes = []
        h_sample = np.arange(0, H, height_separation)
        for dd in data['shapes']:
            plist = np.array(dd['points'])
            mask, line = genmaps(H, W, plist)
            ml = mask & line
            rows, cols = np.nonzero(ml)
            lane = np.ones_like(h_sample) * -2
            lane[rows//10] = cols
            lanes.append(lane.tolist())

        #Label file informations. (TuSimple format)
        info = {}
        info["lanes"] = lanes
        info["raw_file"] = os.path.join(root_folder,  filePath)
        info["h_samples"] = h_sample.tolist()

        json_f.write(json.dumps(info))
        json_f.write('\n')
    
    json_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        required=True,
                        help='The root of the labelme annotated dataset.')
    parser.add_argument('--out_file',
                        type=str,
                        required=True,
                        help='The output json file.')
    args = parser.parse_args()

    create_anno(args=args)