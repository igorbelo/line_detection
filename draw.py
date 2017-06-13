import sys
import os
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint
from matplotlib.patches import Rectangle

class Colormap:
    def __init__(self, n, *args, **kwargs):
        self.colors = ['#ffffff'] + (['#ff0000','#2200ff','#089f08','#000000','#f7d308','#397364'] * n)
        self.line_colors_count = len(self.colors)-1
        self.cmap = mpl.colors.ListedColormap(self.colors)

def draw_best_partition(partition, graph, regions):
    centroids = {i: region.centroid[::-1] for i, region in enumerate(regions)}
    size = float(len(set(partition.values())))
    count = 0.
    colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in set(partition.values())]
    for i, com in enumerate(set(partition.values())):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
        nx.draw_networkx_nodes(graph, centroids, list_nodes, node_size = 1,
                               node_color = colors[i])

    nx.draw_networkx_edges(graph, centroids, edge_color='#ff0000', alpha=0.1, width=1)

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

def generate_step_images(image, binary_image, segmented_image, graph, partition, regions):
    path = "steps/%s" % sys.argv[1]
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imshow(binary_image, cmap='gray')
    plt.savefig("%s/2-binary-image.png" % path, format='png', dpi=300)
    plt.clf()
    plt.imshow(segmented_image)
    plt.savefig("%s/3-segmented-image.png" % path, format='png', dpi=300)
    plt.clf()
    plt.imshow(image)
    plt.savefig("%s/1-original-image.png" % path, format='png', dpi=300)
    draw_best_partition(partition, graph, regions)
    plt.axis('off')
    plt.savefig("%s/4-graph.png" % path, format='png', dpi=300)

def result(segmented_image, communities):
    n = max(communities.values())
    colormap = Colormap(n)

    img2 = segmented_image.copy()
    for label, community in communities.iteritems():
        img2[img2 == label+1] = community+1

    plt.axis('off')
    plt.imshow(img2, vmin=0, vmax=len(colormap.colors), cmap=colormap.cmap)
    plt.savefig("results/%s.png" % sys.argv[1], format='png', dpi=400)
