import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import measure, color, morphology, img_as_ubyte
from skimage.io import imread
from skimage.filters import threshold_adaptive
from scipy.spatial.distance import pdist, squareform

import community
import sys
from ground_truth_reader import build_line_meta
import draw
import cv2

def measure_labels(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32)
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    segmented_image, num_labels = measure.label(
        binary_adaptive, background=1, return_num=True)

    # other_image = morphology.remove_small_objects(segmented_image, 10)
    # segmented_other_image = measure.label(other_image, background=0)

    return (binary_adaptive, (segmented_image, num_labels))

def generate_distances_array(regions):
    centroids_y = np.array([props.centroid[0] for props in regions])
    distances = pdist(centroids_y[:, np.newaxis], metric='cityblock')

    return distances

def generate_adjacency_matrix(distances):
    Y_DISTANCE = np.percentile(distances, 5)
    adjacency_matrix = squareform(distances)
    adjacency_matrix[adjacency_matrix > Y_DISTANCE] = 0

    return adjacency_matrix

def create_mask(binary_image):
    return np.full_like(binary_image_uint, 255, dtype=np.uint8)

def put_component_on_mask(mask, coords):
    y, x = np.hsplit(coords, 2)
    mask[y, x] = 0

def distance_between_two_components(d1, c2, c2_coords):
    y, x = np.hsplit(c2_coords, 2)
    return min(d1[y, x])

def create_nodes(graph, regions):
    for prop in regions:
        graph.add_node(prop.label, pos=prop.centroid)

def generate_graph(distances, regions):
    G = nx.Graph()
    create_nodes(G, regions)
    flat_distances = [item for sublist in distances for item in sublist]
    threshold = np.percentile(flat_distances, 1.5)
    for i, props in enumerate(regions[:-1]):
        distances_from_component = distances[i]
        for j, d in enumerate(distances_from_component, i+1):
            if d <= threshold:
                G.add_edge(i+1, j+1)

    return G

def get_distances_between_components(binary_image, regions):
    distances = []

    for i, props in enumerate(regions):
        c1 = create_mask(binary_image)
        put_component_on_mask(c1, props.coords)
        d1 = cv2.distanceTransform(c1, cv2.NORM_L2,
                                   cv2.DIST_MASK_PRECISE)

        distances.append([])

        for props2 in regions[i+1:]:
            c2 = create_mask(binary_image)
            put_component_on_mask(c2, props2.coords)
            d = distance_between_two_components(d1, c2, props2.coords)
            distances[i].append(d)

    return distances

def calculate_accuracy(ground_truth, regions, communities):
    total = max(communities.values())
    total_lines = len(ground_truth)
    total_components = len(regions)
    found = 0
    for label, community in communities.iteritems():
        if community < total_lines:
            line = ground_truth[community]
            y_top, x_left, y_bottom, x_right = regions[label].bbox
            if y_top >= line["yTop"] and y_bottom <= line["yBottom"] and\
               x_left >= line["xLeft"] and x_right <= line["xRight"]:
                found += 1

    print (float(found) / total_components) * 100.0



if __name__ == '__main__':
    # base_file_name = sys.argv[1]
    base_dir = sys.argv[1]
    for i in range(4, 5):
        filename = '%s-%s' % (base_dir, str(i).zfill(3))
        image = imread("images/%s/%s.png" % (base_dir, filename))
        ground_truth = build_line_meta("ground-truth/%s/%s.json" % (base_dir, filename))
        binary_image, segment = measure_labels(image)
        segmented_image, _ = segment
        binary_image_uint = img_as_ubyte(binary_image)
        regions = measure.regionprops(segmented_image)
        distances = get_distances_between_components(
            binary_image_uint, regions)

        # distances = generate_distances_array(regions)
        # adjacency_matrix = generate_adjacency_matrix(distances)

        # G = nx.from_numpy_matrix(adjacency_matrix)
        if distances:
            G = generate_graph(distances, regions)
            communities = community.best_partition(G)
            # result = calculate_accuracy(ground_truth, regions, communities)
            draw.result(segmented_image, communities, 'results/%s' % base_dir, filename+'.png', 'png')
            draw.generate_step_images(image, binary_image, segmented_image, G, communities, regions, 'steps/%s' % filename)

        plt.clf()
