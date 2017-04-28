import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure, color, morphology
from skimage.io import imread
from skimage.filters import threshold_adaptive
import glob, os, sys, json, itertools
from distutils.dir_util import mkpath
from os.path import basename

def extract_connected_components(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32)
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    segmented_image, num_labels = measure.label(
        binary_adaptive, background=1, return_num=True
    )

    clean_image = morphology.remove_small_objects(segmented_image, 5)
    segmented_clean_image = measure.label(clean_image, background=0)

    return segmented_clean_image

class Extractor:
    def __init__(self, segmented_image, *args, **kwargs):
        self.start_y = None
        self.end_y = None
        self.lines = []
        self.segmented_image = segmented_image

    def set_region(self, clicked_y):
        if self.start_y is None:
            self.start_y = clicked_y
        else:
            self.end_y = clicked_y

    def store_line_boundaries(self, event):
        if event.button == 3:
            clicked_y = event.ydata.astype(int)
            self.set_region(clicked_y)
            if self.start_y and self.end_y:
                min_x, min_y, max_x, max_y = self.get_line_boundaries()
                line = {
                    "yTop": min_y+self.start_y,
                    "yBottom": max_y+self.start_y,
                    "xLeft": min_x,
                    "xRight": max_x
                }
                self.lines.append(line)
                self.start_y = self.end_y
                print "LINHA ADICIONADA"
                print line

    def get_line_boundaries(self):
        regions_props = measure.regionprops(
            self.segmented_image[self.start_y:self.end_y])
        min_y = None
        min_x = None
        max_y = 0
        max_x = 0

        for prop in regions_props:
            bbox = prop.bbox
            if bbox[0] < min_y or min_y is None:
                min_y = bbox[0]
            if bbox[1] < min_x or min_x is None:
                min_x = bbox[1]
            if bbox[2] > max_y:
                max_y = bbox[2]
            if bbox[3] > max_x:
                max_x = bbox[3]

        print (min_x, min_y, max_x, max_y)
        return (min_x, min_y, max_x, max_y)

base_dir = sys.argv[1]
if len(sys.argv) > 2:
    file_pattern = sys.argv[2]
else:
    file_pattern = '*'
filenames = glob.glob('images/%s/%s.png' % (base_dir, file_pattern))
for filename in filenames:
    ground_truth_path = 'ground-truth/%s' % base_dir
    base_filename = basename(os.path.splitext(filename)[0])
    image = imread(filename)
    segmented_image = extract_connected_components(image)
    fig = plt.figure()
    extractor = Extractor(segmented_image)

    plt.imshow(image)
    fig.canvas.mpl_connect('button_press_event', extractor.store_line_boundaries)
    plt.show()
    plt.close()
    mkpath(ground_truth_path)

    ground_truth_filename = '%s/%s.json' % (ground_truth_path, base_filename)
    if extractor.lines or not os.path.isfile(ground_truth_filename) or os.path.getsize(ground_truth_filename) == 0:
        with open(ground_truth_filename, 'w+') as f:
            f.write(json.dumps({"lines": extractor.lines}))
