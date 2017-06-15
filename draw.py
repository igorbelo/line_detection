import sys
import os
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
from matplotlib.patches import Rectangle
import itertools
from skimage import measure

class Colormap:
    def __init__(self, n, *args, **kwargs):
        self.colors = ['#ffffff'] + (['#ff0000','#2200ff','#089f08','#000000','#f7d308','#397364'] * n)
        self.line_colors_count = len(self.colors)-1
        self.cmap = mpl.colors.ListedColormap(self.colors)

def draw_best_partition(partition, graph, regions):
    n = max(partition.values())
    colormap = Colormap(n)
    centroids = {region.label: region.centroid[::-1] for region in regions}
    size = float(len(set(partition.values())))
    colors = colormap.colors
    for community, nodes in itertools.groupby(
            partition,
            lambda item: partition[item]):
        nx.draw_networkx_nodes(graph, centroids, list(nodes), node_size = 1,
                               node_color = colors[community+1])

def fill_lines(partition, regions):
    colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in range(len(set(partition.values())))]

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

    n = max(communities.values())
    colormap = Colormap(n)

    img2 = segmented_image.copy()
    regions = measure.regionprops(img2)
    for label, community in communities.iteritems():
        img2[img2 == label] = community+1

    plt.axis('off')
    plt.imshow(img2, vmin=0, vmax=len(colormap.colors), cmap=colormap.cmap)
    plt.savefig("%s/%s" % (write_on, filename), format=format, dpi=400)
