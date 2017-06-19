import sys
import os
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
from matplotlib.patches import Rectangle
import itertools
from skimage import measure
import numpy as np

hex_colors = ['#ff0000','#2200ff','#089f08','#000000','#f7d308','#397364']
rgb_colors = [(255,0,0),(34,0,255),(8,159,8),(0,0,0),(247,211,8),(57,115,100)]

def draw_best_partition(partition, graph, regions):
    centroids = {region.label: region.centroid[::-1] for region in regions}
    for community, nodes in itertools.groupby(
            partition,
            lambda item: partition[item]):
        nx.draw_networkx_nodes(graph, centroids, list(nodes), node_size = 1,
                               node_color = hex_colors[community % len(hex_colors)])

def fill_lines(segmented_img, rgb_img, communities):
    for label, community in communities.iteritems():
        rgb_img[segmented_img == label] = rgb_colors[community % len(rgb_colors)]


def generate_step_images(image, binary_image, segmented_image, graph, partition, regions, write_on):
    if not os.path.exists(write_on):
        os.makedirs(write_on)

    plt.imshow(binary_image, cmap='gray')
    plt.savefig("%s/2-binary-image.png" % write_on, format='png', dpi=300)
    plt.clf()
    plt.imshow(segmented_image)
    plt.savefig("%s/3-segmented-image.png" % write_on, format='png', dpi=300)
    plt.clf()
    plt.imshow(image)
    plt.savefig("%s/1-original-image.png" % write_on, format='png', dpi=300)
    draw_best_partition(partition, graph, regions)
    plt.axis('off')
    plt.savefig("%s/4-graph.png" % write_on, format='png', dpi=300)

def result(segmented_image, communities, write_on, filename, format='png'):
    if not os.path.exists(write_on):
        os.makedirs(write_on)

    rgb = np.ones(segmented_image.shape + (3,), dtype=np.uint8) * 255
    fill_lines(segmented_image, rgb, communities)

    plt.axis('off')
    plt.imshow(rgb)
    plt.savefig("%s/%s" % (write_on, filename), format=format, dpi=400)
