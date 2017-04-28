import matplotlib.pyplot as plt
import glob, os, sys, json, itertools
from distutils.dir_util import mkpath
from os.path import basename
from skimage.io import imread
from distutils.dir_util import mkpath
from itertools import cycle

base_dir = sys.argv[1]
if len(sys.argv) > 2:
    file_pattern = sys.argv[2]
else:
    file_pattern = '*'
filenames = glob.glob('images/%s/%s.png' % (base_dir, file_pattern))
for filename in filenames:
    colors = cycle(['blue', 'red', 'yellow', 'green'])
    ground_truth_path = 'ground-truth/%s' % base_dir
    base_filename = basename(os.path.splitext(filename)[0])
    with open("%s/%s.json" % (ground_truth_path, base_filename)) as f:
        ground_truth = json.loads(f.read())

    image = imread(filename)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    ground_truth_image_path = "images/%s/ground-truth" % base_dir
    mkpath(ground_truth_image_path)
    for line in ground_truth['lines']:
        plt.plot(
            [line['xLeft'], line['xRight']],
            [line['yTop'], line['yBottom']],
            '.',
            color=colors.next(),
            markersize=1
        )
        plt.savefig("%s/%s.eps" % (ground_truth_image_path, base_filename), format='eps', dpi=400)
    plt.clf()
