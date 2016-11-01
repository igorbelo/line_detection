import numpy as np
from skimage import measure, color
import matplotlib.pyplot as plt

from skimage import data
from skimage.io import imread
from skimage.filters import threshold_adaptive
import networkx as nx
from skimage.morphology import square, closing

import community
import sys
import os

Y_DISTANCE = 5

def measure_labels(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32) #convert para escala de cor 0 -255
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    segmented_image = measure.label(binary_adaptive, background=1)

    return (binary_adaptive, segmented_image)

def plot_centroids(image, regions):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        ax.plot(x0, y0, '.g', markersize=5)

    ax.axis((0, len(image[0]), len(image), 0))
    plt.show()

def create_nodes_from_regions(graph, regions):
    for label, prop in enumerate(regions):
        row, col = prop.centroid
        graph.add_node(label, pos=(col, row))

def create_edges_from_regions(graph, regions):
    for label, properties in enumerate(regions):
        [graph.add_edge(label, destination_label) for destination_label, destination_properties in enumerate(regions) if abs(properties.centroid[0] - destination_properties.centroid[0]) <= Y_DISTANCE]

def draw_best_partition(partition, graph):
    #drawing
    pos = nx.get_node_attributes(graph, 'pos')
    size = float(len(set(partition.values())))
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
        nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 20,
                               node_color = str(count / size))

    nx.draw_networkx_edges(graph, pos, edge_color='#ff0000', alpha=0.1, width=1)

def draw_graph(graph):
    nx.draw(graph, nx.get_node_attributes(graph, 'pos'), node_size=25)

def fill_lines(partition):
    from matplotlib.patches import Rectangle

    currentAxis = plt.gca()
    last_max_x = None
    last_community = None
    for element_index, community_index in partition.items():
        if last_community is not None and last_community != community_index:
            last_max_x = None

        prop = regions[element_index]
        min_y, min_x, max_y, max_x = prop.bbox

        if last_max_x is not None:
            min_x = last_max_x

        currentAxis.add_patch(
            Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                alpha=0.1,
                facecolor=colors[community_index],
                edgecolor='none'
            )
        )
        last_max_x = max_x
        last_community = community_index

from skimage import img_as_uint
def generate_step_images(image, binary_image, segmented_image, graph, partition):
    path = "steps/%s" % sys.argv[1]
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imshow(binary_image)
    plt.savefig("%s/2-binary-image.jpg" % path)
    plt.clf()
    plt.imshow(segmented_image)
    plt.savefig("%s/3-segmented-image.jpg" % path)
    plt.clf()
    plt.imshow(image)
    plt.savefig("%s/1-original-image.jpg" % path)
    draw_best_partition(partition, graph)
    plt.savefig("%s/4-graph.jpg" % path)
    plt.clf()
    plt.imshow(image)
    fill_lines(partition)
    plt.savefig("%s/5-partition.jpg" % path)

image = imread("images/%s" % sys.argv[1])
binary_image, segmented_image = measure_labels(image)
regions = measure.regionprops(segmented_image)
G = nx.Graph()
create_nodes_from_regions(G, regions)
create_edges_from_regions(G, regions)
partition = community.best_partition(G)

from random import randint
colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in range(len(set(partition.values())))]
generate_step_images(
    image,
    binary_image,
    segmented_image,
    G,
    partition
)

# print len(regions)
# print partition

# print segmented_image[300]

# plt.imshow(image)
# fill_lines(partition)

# plt.show()
