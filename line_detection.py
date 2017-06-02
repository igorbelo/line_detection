import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import measure, color, morphology
from skimage.io import imread
from skimage.filters import threshold_adaptive
from scipy.spatial.distance import pdist, squareform

import community
import sys
from ground_truth_reader import build_line_meta
import draw

def measure_labels(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32)
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    segmented_image, num_labels = measure.label(
        binary_adaptive, background=1, return_num=True)

    other_image = morphology.remove_small_objects(segmented_image, 20)
    segmented_other_image = measure.label(other_image, background=0)

    return (binary_adaptive, (segmented_other_image, num_labels))

def generate_distances_array(regions):
    centroids_y = np.array([props.centroid[0] for props in regions])
    distances = pdist(centroids_y[:, np.newaxis], metric='cityblock')

    return distances

def generate_adjacency_matrix(distances):
    Y_DISTANCE = np.percentile(distances, 2.5)
    adjacency_matrix = squareform(distances)
    adjacency_matrix[adjacency_matrix > Y_DISTANCE] = 0

    return adjacency_matrix

if __name__ == '__main__':
    base_file_name = sys.argv[1]
    # ground_truth = build_line_meta("ground-truth/%s.json" % base_file_name)
    image = imread("images/%s.png" % base_file_name)
    binary_image, segment = measure_labels(image)
    segmented_image, num_labels = segment
    regions = measure.regionprops(segmented_image)
    distances = generate_distances_array(regions)
    adjacency_matrix = generate_adjacency_matrix(distances)

    G = nx.from_numpy_matrix(adjacency_matrix)
    communities = community.best_partition(G)
    n = max(communities.values())

    colormap = draw.Colormap(n)

    img2 = segmented_image.copy()

    for label, community in communities.iteritems():
        img2[img2 == label+1] = community+1

    plt.axis('off')
    plt.imshow(img2, vmin=0, vmax=len(colormap.colors), cmap=colormap.cmap)
    plt.savefig("results/%s.eps" % sys.argv[1], format='eps', dpi=400)
    draw.generate_step_images(image, binary_image, segmented_image, G, communities, regions)
