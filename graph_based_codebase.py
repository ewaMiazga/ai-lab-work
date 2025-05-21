import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import concurrent.futures
import json
import os

import bezier
from scipy import ndimage as ndi

def preprocess_image(image):
    """Convert an image to binary and skeletonize it."""
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary // 255)
    return skeleton


def extract_graph(skeleton):
    """Extract a well-connected graph from a skeletonized image and center it."""
    G = nx.Graph()
    height, width = skeleton.shape

    # Find all active pixel positions
    pixel_positions = np.argwhere(skeleton == 1)

    if len(pixel_positions) == 0:
        return G  # Return empty graph if no skeleton pixels
    
    pos_dict = {}  # Dictionary to store positions

    # Define 8-neighbor connectivity (left, right, up, down, diagonals)
    neighbors = [
        (-1, 0), (1, 0),  # Left, Right
        (0, -1), (0, 1),  # Up, Down
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal connections
    ]

    for y, x in pixel_positions:
        # Shift node positions so the centroid aligns at (0,0)
        node = (x, -y)  # Convert to tuple for hashing
        G.add_node(node)
        pos_dict[node] = node  # Store position

        # Connect to valid 8-neighbors
        for dx, dy in neighbors:
            nx_pos, ny_pos = x + dx, y + dy
            neighbor_node = nx_pos, -ny_pos

            if 0 <= nx_pos < width and 0 <= ny_pos < height and skeleton[ny_pos, nx_pos] == 1:
                weight = 1 if abs(dx) + abs(dy) == 1 else np.sqrt(2)  # Euclidean distance for diagonals
                G.add_edge(node, neighbor_node, weight=weight)

    # Assign positions to nodes
    nx.set_node_attributes(G, pos_dict, 'pos')

    return G

def plot_graph(G, ax):
    """Plot the graph on the given axis."""
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, ax=ax, node_size=5, edge_color='gray', with_labels=False)


def merge_similar_edges(G, angle_threshold=10):
    """
    Merges edges in the graph that have nearly the same direction.
    Stores the length and original position of each merged edge.
    
    :param G: Input graph with node positions.
    :param angle_threshold: Maximum angle (in degrees) for merging.
    :return: A new graph with merged edges.
    """
    merged_G = nx.Graph()
    pos = nx.get_node_attributes(G, 'pos')  # Get the original positions of the nodes

    # Convert graph edges into vectors (origin, direction)
    edge_vectors = {}
    for u, v in G.edges():
        origin = np.array(pos[u])
        direction = np.array(pos[v]) - np.array(pos[u])
        edge_vectors[(u, v)] = (origin, direction)

    # Function to compute angle between two vectors
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Ensure valid range

    visited = set()
    
    for (u, v), (origin, direction) in edge_vectors.items():
        if (u, v) in visited or (v, u) in visited:
            continue
        
        current_path = [(u, v)]  # Store the edges being merged
        total_length = np.linalg.norm(direction)
        merged_origin = origin
        merged_direction = direction
        
        # Try to extend the merged path
        while True:
            next_edge = None
            for neighbor in G.neighbors(v):
                if neighbor == u:  # Avoid going backward
                    continue
                if (v, neighbor) in edge_vectors:
                    _, next_dir = edge_vectors[(v, neighbor)]
                else:
                    continue

                angle = angle_between(merged_direction, next_dir)
                if angle < angle_threshold:
                    next_edge = (v, neighbor)
                    break  # Merge the first valid edge

            if next_edge:
                _, next_dir = edge_vectors[next_edge]
                total_length += np.linalg.norm(next_dir)
                merged_direction += next_dir  # Accumulate the direction
                current_path.append(next_edge)
                visited.add(next_edge)
                u, v = next_edge  # Move forward
            else:
                break  # No more similar edges to merge

        # Add merged edge to new graph
        new_node = tuple(merged_origin)  # Store as tuple for Graph
        new_endpoint = tuple(merged_origin + merged_direction)  # End of merged edge
        merged_G.add_edge(new_node, new_endpoint, length=total_length, original_edges=current_path)

        # Set positions for new nodes
        merged_G.nodes[new_node]['pos'] = merged_origin  # Position of merged node
        merged_G.nodes[new_endpoint]['pos'] = merged_origin + merged_direction  # Position of endpoint of the merged edge

    # Transfer the original positions of existing nodes from the original graph
    for node in G.nodes():
        if node in merged_G.nodes:
            merged_G.nodes[node]['pos'] = pos[node]
    
    return merged_G


def detect_curvatures(merged_G, curvature_threshold=5, min_length=1):
    """
    Detects curvature points in the merged graph based on angle changes.
    
    :param merged_G: Graph with merged edges (from previous step).
    :param curvature_threshold: Angle (degrees) above which we classify as curvature.
    :param min_length: Minimum length of a segment to consider for curvature detection.
    :return: List of curvature points.
    """
    curvature_points = []
    
    # Extract edges and their directions
    edges = list(merged_G.edges(data=True))
    
    def compute_angle(v1, v2):
        """Computes the angle (in degrees) between two vectors."""
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to valid range

    for i in range(len(edges) - 1):
        (u1, v1, data1) = edges[i]
        (u2, v2, data2) = edges[i + 1]

        # Compute direction vectors
        direction1 = np.array(v1) - np.array(u1)
        direction2 = np.array(v2) - np.array(u2)

        # Compute angle change
        angle_change = compute_angle(direction1, direction2)

        # If the angle change is significant, mark it as a curvature
        if angle_change > curvature_threshold and data1["length"] > min_length and data2["length"] > min_length:
            curvature_points.append(v1)  # Store the node where curvature happens

    return curvature_points

def extract_long_paths(G, min_length=3):
    visited = set()
    paths = []

    for node in G.nodes:
        if G.degree[node] == 1:  # endpoint
            path = [node]
            current = node
            prev = None
            while True:
                neighbors = list(G.neighbors(current))
                next_nodes = [n for n in neighbors if n != prev]
                if not next_nodes or next_nodes[0] in visited:
                    break
                prev = current
                current = next_nodes[0]
                path.append(current)
            if len(path) >= min_length:
                paths.append(path)
                visited.update(path)
    return paths

# Function to fit a Bézier curve to a given path
def fit_bezier_curve(path):
    # Extract x and y coordinates of the path
    x = np.array([node[0] for node in path])
    y = np.array([node[1] for node in path])
    
    # Create a Bézier curve using the coordinates (you can try fitting a higher degree)
    nodes = np.asfortranarray([x, y])
    curve = bezier.Curve(nodes, degree=len(nodes[0]) - 1)

    # Pack control points as a list of [x, y] pairs
    control_points = [[float(x[i]), float(y[i])] for i in range(len(x))]
    return curve, control_points

# Function to generate Bézier curve points for visualization
def get_bezier_curve_points(curve, num_points=100):
    # Generate points along the Bézier curve
    u_vals = np.linspace(0.0, 1.0, num_points)
    curve_points = curve.evaluate_multi(u_vals)
    return curve_points[0], curve_points[1]  # x and y


# Function to identify and display global and local curves for different sensitivity levels
def identify_and_display_curves(G, min_length_values=[3, 2, 1]):
    fig, axes = plt.subplots(1, len(min_length_values), figsize=(15, 5))
    axes = axes.flatten()

    for i, min_length in enumerate(min_length_values):
        paths = extract_long_paths(G, min_length=min_length)

        # Plot the graph with detected curves for the current min_length
        ax = axes[i]
        nx.draw(G, pos={node: node for node in G.nodes()}, node_size=5, edge_color='lightgray', ax=ax)

        # Loop through the paths and fit Bézier curves
        for path in paths:
            curve, control_points = fit_bezier_curve(path)
            bezier_x, bezier_y = get_bezier_curve_points(curve)

            # Use different colors for each level of sensitivity (min_length)
            ax.plot(bezier_x, bezier_y, label=f'Curves (min_length={min_length})', linewidth=2)

        ax.set_title(f"Curves Detected at min_length={min_length}")
        ax.legend()

    plt.tight_layout()
    plt.show()


# --- Bézier utilities ---

def extract_long_paths_fast(G, min_length=3):
    visited = set()
    paths = []

    for node in G.nodes:
        if node in visited or G.degree[node] != 1:
            continue  # only start from endpoints not already visited

        path = [node]
        current = node
        prev = None
        while True:
            neighbors = list(G.neighbors(current))
            next_nodes = [n for n in neighbors if n != prev]
            if not next_nodes or next_nodes[0] in visited or len(next_nodes) > 1:
                break
            prev = current
            current = next_nodes[0]
            path.append(current)

        if len(path) >= min_length:
            paths.append(path)
            visited.update(path)  # mark entire path after we know it’s valid
    return paths

def fit_bezier_curve_fast(path):
    x = np.array([node[0] for node in path])
    y = np.array([node[1] for node in path])
    if len(x) < 2:
        return None  # Too few points
    degree = min(len(x) - 1, 5)  # Limit max degree to avoid instability
    nodes = np.asfortranarray([x, y])
    return bezier.Curve(nodes[:, :degree + 1], degree=degree)

def plot_graph_with_beziers_multigraph(G, ax, color ="red"):
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, node_size=1, edge_color='gray', alpha=0.2, ax=ax)

    paths = extract_long_paths_fast(G)
    for path in paths:
        curve = fit_bezier_curve_fast(path)
        if curve:
            bx, by = get_bezier_curve_points(curve)
            ax.plot(bx, by, color=color, linewidth=1.5)


def detect_almost_circles(G, tolerance=0.05, min_length=4):
    """
    Detect "almost circles" in a graph based on geometric properties.
    
    :param G: Input graph.
    :param tolerance: Tolerance for detecting "almost circular" cycles.
    :return: List of "almost circular" cycles.
    """
    cycles = nx.cycle_basis(G)  # Detect cycles in the graph
    almost_circles = []

    for cycle in cycles:
        # Get the positions of nodes in the cycle
        pos = nx.get_node_attributes(G, 'pos')
        cycle_positions = np.array([pos[node] for node in cycle])

        # Compute the centroid of the cycle
        centroid = np.mean(cycle_positions, axis=0)

        # Compute distances of nodes from the centroid
        distances = np.linalg.norm(cycle_positions - centroid, axis=1)

        # Check if distances are approximately equal (within tolerance)
        # And the number of nodes in the cycle is greater than min_length
        if np.std(distances) / np.mean(distances) < tolerance and len(cycle) > min_length:
            almost_circles.append(cycle)

    return almost_circles

def plot_almost_circles(G, almost_circles, ax):
    """
    Plot the graph and highlight "almost circular" cycles.
    
    :param G: Input graph.
    :param almost_circles: List of "almost circular" cycles.
    :param ax: Matplotlib axes to plot on.
    """
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=5, edge_color='gray', alpha=0.5, ax=ax)

    for cycle in almost_circles:
        cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)] + [(cycle[-1], cycle[0])]
        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='green', width=2, ax=ax)

    ax.axis('off')  # Hide axis