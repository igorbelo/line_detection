import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure, color, morphology
from skimage.io import imread
from skimage.filters import threshold_adaptive
import glob, os, sys, json, itertools
from distutils.dir_util import mkpath
from os.path import basename
import math

SPLIT_SIZE = 30

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

    clean_image = morphology.remove_small_objects(segmented_image, 25)
    segmented_clean_image = measure.label(clean_image, background=0)

    return segmented_clean_image

class Extractor:
    def __init__(self, segmented_image, *args, **kwargs):
        self.y_clicks = []
        self.lines = []
        self.splitted_lines = {}
        self.segmented_image = segmented_image
        self.width = len(segmented_image[0])
        self.height = len(segmented_image)

    def on_click(self, event):
        if event.button == 3:
            clicked_y = event.ydata.astype(int)
            self.y_clicks.append(clicked_y)
            if len(self.y_clicks) > 1:
                y_limits = self.get_y_click_range()
                boundaries = self.get_line_boundaries(y_limits=y_limits)
                self.store_line_boundaries(boundaries)

    def get_line_boundaries(self, **kwargs):
        x_limits = kwargs.get('x_limits', None)
        y_limits = kwargs.get('y_limits', None)

        if x_limits:
            start_x, end_x = x_limits
        else:
            start_x, end_x = (0, self.width-1)

        if y_limits:
            start_y, end_y = y_limits
        else:
            start_y, end_y = (0, self.height-1)

        regions_props = measure.regionprops(
            self.segmented_image[start_y:end_y, start_x:end_x])
        min_y, min_x, max_y, max_x = [None]*4

        for prop in regions_props:
            bbox = prop.bbox
            if bbox[0] < min_y or min_y is None:
                min_y = bbox[0]
            if bbox[1] < min_x or min_x is None:
                min_x = bbox[1]
            if bbox[2] > max_y or max_y is None:
                max_y = bbox[2]
            if bbox[3] > max_x or max_x is None:
                max_x = bbox[3]

        return (min_x, min_y, max_x, max_y)

    def create_line(self, boundaries, start_y=None, end_y=None):
        if not start_y:
            start_y = self.get_start_y()
        if not end_y:
            end_y = self.get_end_y()
        min_x, min_y, max_x, max_y = boundaries
        return {
            "yTop": min_y+start_y,
            "yBottom": max_y+start_y,
            "xLeft": min_x,
            "xRight": max_x
        }

    def store_line_boundaries(self, boundaries):
        if None not in boundaries:
            line = self.create_line(boundaries)
            self.lines.append(line)
            self.start_y = self.get_end_y()
            print "LINHA ADICIONADA"
            print line
            print self.y_clicks

    def get_y_click_range(self):
        return self.y_clicks[-2:]

    def get_start_y(self):
        return self.y_clicks[-2]

    def get_end_y(self):
        return self.y_clicks[-1]

    def get_split_width(self, acc, counter):
        max_split_width = math.ceil(self.width / float(SPLIT_SIZE))
        if (self.width - acc) % (SPLIT_SIZE - counter) == 0:
            return (self.width - acc) / (SPLIT_SIZE - counter)
        else:
            return max_split_width


    def split(self):
        acc = 0
        counter = 0
        for i in range(SPLIT_SIZE):
            split_width = self.get_split_width(acc, counter)

            last_y = None
            self.splitted_lines[i] = []
            for y in self.y_clicks:
                if last_y:
                    y_limits = (last_y, y)
                    boundaries = self.get_line_boundaries(y_limits=y_limits, x_limits=(acc, acc+split_width-1))
                    if not None in boundaries:
                        line = self.create_line(boundaries, y_limits[0], y_limits[1])
                        self.splitted_lines[i].append(line)
                last_y = y
            acc += split_width
            counter += 1

class GroundTruth:
    def __init__(self, extractor, base_dir, filename, *args, **kwargs):
        self.extractor = extractor
        self.ground_truth_path = 'ground-truth/%s' % base_dir
        self.base_filename = basename(os.path.splitext(filename)[0])
        self.output_filename = '%s/%s.json' % (self.ground_truth_path, self.base_filename)

    def write(self):
        mkpath(self.ground_truth_path)
        if self.extractor.lines or \
           not os.path.isfile(self.output_filename) or \
           os.path.getsize(self.output_filename) == 0:
            with open(self.output_filename, 'w+') as f:
                f.write(json.dumps({"lines": self.extractor.lines}))

            for i in range(SPLIT_SIZE):
                output_filename = "%s/%s-%s.json" % (self.ground_truth_path, self.base_filename, str(i).zfill(3))
                with open(output_filename, 'w+') as f:
                    f.write(json.dumps({"lines": self.extractor.splitted_lines[i]}))


base_dir = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2] == '--auto-split':
        auto_split = True
        file_pattern = base_dir
    else:
        auto_split = False
        file_pattern = sys.argv[2]
else:
    file_pattern = '*'

filenames = glob.glob('images/%s/%s.png' % (base_dir, file_pattern))
for filename in filenames:
    image = imread(filename)
    segmented_image = extract_connected_components(image)
    fig = plt.figure()
    plt.imshow(image)
    extractor = Extractor(segmented_image)
    ground_truth = GroundTruth(extractor, base_dir, filename)
    fig.canvas.mpl_connect('button_press_event', extractor.on_click)
    plt.show()
    plt.close()
    if auto_split:
        extractor.split()
    ground_truth.write()
