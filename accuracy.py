import sys
from skimage import measure, color
from skimage.io import imread
from skimage.filters import threshold_adaptive
import numpy as np
import json
import itertools
import cv2

MATCH_THRESHOLD = 0.8

def get_segmented_image(img_file):
    img = imread(img_file)
    work_img = (255*color.rgb2gray(img)).astype(np.int32)
    binary_image = threshold_adaptive(work_img, 41, offset=10)
    return measure.label(binary_image, background=1)

def load_ground_truth_lines(file):
    with open(file) as f:
        return json.loads(f.read())['lines']

def get_bbox(components):
    for component in components:
        if 'y_top' not in vars():
            y_top = component["y_top"]
            y_bottom = component["y_bottom"]
            x_left = component["x_left"]
            x_right = component["x_right"]
            continue
        if component["y_top"] < y_top:
            y_top = component["y_top"]
        if component["y_bottom"] > y_bottom:
            y_bottom = component["y_bottom"]
        if component["x_left"] < x_left:
            x_left = component["x_left"]
        if component["x_right"] > x_right:
            x_right = component["x_right"]

    return {"y_top": y_top, "x_left": x_left, "y_bottom": y_bottom,
            "x_right": x_right}


def load_hypothesis_lines(file):
    lines = []
    with open(file) as f:
        components = json.loads(f.read())
    grouped_lines = itertools.groupby(components, lambda d: d['community'])

    for _, components in grouped_lines:
        components = list(components)
        lines.append(get_bbox(components))

    return lines

def create_h_mask(image, line_bbox):
    mask = np.full_like(image, 0, dtype=np.uint8)
    mask[line_bbox["y_top"]:line_bbox["y_bottom"]+1, line_bbox["x_left"]:line_bbox["x_right"]+1] = image[line_bbox["y_top"]:line_bbox["y_bottom"]+1, line_bbox["x_left"]:line_bbox["x_right"]+1]
    return mask

def create_r_mask(image, line_bbox):
    mask = np.full_like(image, 0, dtype=np.uint8)
    mask[line_bbox["yTop"]:line_bbox["yBottom"]+1, line_bbox["xLeft"]:line_bbox["xRight"]+1] = image[line_bbox["yTop"]:line_bbox["yBottom"]+1, line_bbox["xLeft"]:line_bbox["xRight"]+1]
    return mask

def match_score(h_mask, r_mask):
    N_inter = (cv2.bitwise_and(h_mask,r_mask) > 0).sum()
    N_union = (cv2.bitwise_or(h_mask,r_mask) > 0).sum()
    return float(N_inter) / N_union

# filename = sys.argv[1]
segmented_image = get_segmented_image("images/undersea016/undersea016-006.png")

R_lines = load_ground_truth_lines("ground-truth/undersea016/undersea016-006.json")
H_lines = load_hypothesis_lines("results/undersea016/undersea016-006.json")

matches = 0
for h_line in H_lines:
    h_mask = create_h_mask(segmented_image, h_line)

    for r_line in R_lines:
        r_mask = create_r_mask(segmented_image, r_line)

        if match_score(h_mask, r_mask) >= MATCH_THRESHOLD:
            matches += 1

DR = float(matches) / len(R_lines)
RA = float(matches) / len(H_lines)

error = 1 - (((2 * DR * RA)) / (DR + RA))
