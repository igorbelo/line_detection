import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage import data, measure, color
from skimage.io import imread
from skimage.filters import threshold_adaptive
from scipy.spatial.distance import pdist, squareform

import community
import sys
import os
from random import randint

def measure_labels(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32)
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    segmented_image = measure.label(binary_adaptive, background=1)

    return (binary_adaptive, segmented_image)

# def create_nodes_from_regions(graph, regions):
#     """add nodes to graph based on centroid of each segmented element (region)"""
#     [graph.add_node(label, pos=(prop.centroid[1], prop.centroid[0])) for label, prop in enumerate(regions)]

# def create_edges_from_regions(graph, regions):
#     """tie the nodes if the y distance between the nodes is lower than Y_DISTANCE"""
#     for label, properties in enumerate(regions):
#         [graph.add_edge(label, destination_label) for destination_label, destination_properties in enumerate(regions) if abs(properties.centroid[0] - destination_properties.centroid[0]) <= Y_DISTANCE]

# def draw_best_partition(partition, graph):
#     pos = nx.get_node_attributes(graph, 'pos')
#     size = float(len(set(partition.values())))
#     count = 0.
#     colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in set(partition.values())]
#     for i, com in enumerate(set(partition.values())):
#         count = count + 1.
#         list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#         nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 20,
#                                node_color = colors[i])

#     nx.draw_networkx_edges(graph, pos, edge_color='#ff0000', alpha=0.1, width=1)

# def draw_graph(graph):
#     nx.draw(graph, nx.get_node_attributes(graph, 'pos'), node_size=25)

# def fill_lines(partition):
#     colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in range(len(set(partition.values())))]

#     currentAxis = plt.gca()
#     last_max_x = None
#     last_community = None
#     for element_index, community_index in partition.items():
#         if last_community is not None and last_community != community_index:
#             last_max_x = None

#         prop = regions[element_index]
#         min_y, min_x, max_y, max_x = prop.bbox

#         if last_max_x is not None:
#             min_x = last_max_x

#         currentAxis.add_patch(
#             Rectangle(
#                 (min_x, min_y),
#                 max_x - min_x,
#                 max_y - min_y,
#                 alpha=0.1,
#                 facecolor=colors[community_index],
#                 edgecolor='none'
#             )
#         )
#         last_max_x = max_x
#         last_community = community_index

# def generate_step_images(image, binary_image, segmented_image, graph, partition):
#     path = "steps/%s" % sys.argv[1]
#     if not os.path.exists(path):
#         os.makedirs(path)

#     plt.imshow(binary_image, cmap='gray')
#     plt.savefig("%s/2-binary-image.jpg" % path)
#     plt.clf()
#     plt.imshow(segmented_image)
#     plt.savefig("%s/3-segmented-image.jpg" % path)
#     plt.clf()
#     plt.imshow(image)
#     plt.savefig("%s/1-original-image.jpg" % path)
#     # draw_best_partition(partition, graph)
#     plt.savefig("%s/4-graph.jpg" % path)
#     plt.clf()
#     plt.imshow(image)
#     fill_lines(partition)
#     plt.savefig("%s/5-partition.jpg" % path)

def generate_distances_array(regions):
    centroids_y = np.array([props.centroid[0] for props in regions])
    distances = pdist(centroids_y[:, np.newaxis], metric='cityblock')

    return distances

def generate_adjacency_matrix(distances):
    Y_DISTANCE = np.percentile(distances, 5)
    adjacency_matrix = squareform(distances)
    adjacency_matrix[adjacency_matrix > Y_DISTANCE] = 0

    return adjacency_matrix

class Colormap:
    def __init__(self, *args, **kwargs):
        self.colors = ['#ffffff','#ff0000','#2200ff','#089f08','#000000']
        self.line_colors_count = len(self.colors)-1
        self.cmap = mpl.colors.ListedColormap(self.colors)

    def translate_color(self, n):
        return (n % colormap.line_colors_count) + 1

if __name__ == '__main__':
    image = imread("images/%s" % sys.argv[1])
    binary_image, segmented_image = measure_labels(image)
    regions = measure.regionprops(segmented_image)
    distances = generate_distances_array(regions)
    adjacency_matrix = generate_adjacency_matrix(distances)

    G = nx.from_numpy_matrix(adjacency_matrix)
    communities = community.best_partition(G)

    colormap = Colormap()

    img2 = segmented_image.copy()

    for label, community in communities.iteritems():
        img2[img2 == label+1] = colormap.translate_color(community)

    img2[img2 == 0] = 0

    plt.imshow(img2, vmin=0, vmax=len(colormap.colors), cmap=colormap.cmap)
    plt.savefig("results/%s" % sys.argv[1])
