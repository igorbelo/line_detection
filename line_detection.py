import numpy as np
from skimage import measure, color
import matplotlib.pyplot as plt

from skimage import data
from skimage.io import imread
from skimage.filters import threshold_adaptive
import networkx as nx
from skimage.morphology import square, closing

import community

Y_DISTANCE = 5

def measure_labels(image):
    if image.ndim > 2:
        work_image = (255*color.rgb2gray(image)).astype(np.int32) #convert para escala de cor 0 -255
    else:
        work_image = image

    block_size = 41
    binary_adaptive = threshold_adaptive(work_image, block_size, offset=10)
    # binary_adaptive = closing(binary_adaptive, square(3))
    segmented_image = measure.label(binary_adaptive, background=1)

    return segmented_image

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
        graph.add_node(label, pos=prop.centroid)

def create_edges_from_regions(graph, regions):
    for label, properties in enumerate(regions):
        [graph.add_edge(label, destination_label) for destination_label, destination_properties in enumerate(regions) if abs(properties.centroid[1] - destination_properties.centroid[1]) <= Y_DISTANCE]

def draw_best_partition(graph):
    partition = community.best_partition(graph)

    #drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(graph)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
        nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 20,
                           node_color = str(count / size))

    nx.draw_networkx_edges(graph, pos, alpha=0.5)

def draw_graph(graph):
    nx.draw(graph, nx.get_node_attributes(graph, 'pos'), node_size=50, font_size=9)

# image = data.page()
image = imread("images/1.jpeg")
segmented_image = measure_labels(image)
regions = measure.regionprops(segmented_image)
G = nx.Graph()
create_nodes_from_regions(G, regions)
create_edges_from_regions(G, regions)
# draw_graph(G)
draw_best_partition(G)
plt.show()
