from PIL import Image
import multiprocessing as mp
import colorsys
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Helper functions
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h * 360
    s = s * 100
    v = v * 100
    return h, s, v

def is_similar_color(h1, s1, v1, h2, s2, v2, h_range=5, s_range=5, v_range=5):
    return abs(h1 - h2) <= h_range and abs(s1 - s2) <= s_range and abs(v1 - v2) <= v_range

def update_cluster_average(cluster):
    num_colors = len(cluster)
    total_h = total_s = total_v = 0
    for i in range(num_colors):
        total_h += cluster[i][2][0]
        total_s += cluster[i][2][1]
        total_v += cluster[i][2][2]
    avg_h = total_h / num_colors
    avg_s = total_s / num_colors
    avg_v = total_v / num_colors
    return avg_h, avg_s, avg_v

def process_pixel(args):
    x, y, rgb_image = args
    r, g, b = rgb_image.getpixel((x, y))
    return x, y, rgb_to_hsv(r, g, b)

def split_connected_components(cluster):
    xs, ys = zip(*[(x, y) for x, y, hsv in cluster])
    width, height = max(xs) + 1, max(ys) + 1

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, hsv in cluster:
        binary_mask[y, x] = 1

    labels = np.zeros_like(binary_mask)

    clusters_dict = {}
    for (x, y, hsv), label in zip(cluster, labels[binary_mask == 1]):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append((x, y, hsv))

    clusters_list = list(clusters_dict.values())

    return clusters_list

def load_and_cluster_image(image_path):
    image = Image.open(image_path)
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    pool = mp.Pool(mp.cpu_count())
    hsv_values = pool.map(process_pixel, [(x, y, rgb_image) for x in range(width) for y in range(height)])
    pool.close()
    pool.join()

    hsv_array = [[None for _ in range(width)] for _ in range(height)]
    for x, y, hsv in hsv_values:
        hsv_array[y][x] = (x, y, hsv)

    clusters = []
    cluster_averages = []

    for y in range(height):
        for x in range(width):
            x_coord, y_coord, hsv = hsv_array[y][x]
            h, s, v = hsv
            found_cluster = False
            for i in range(len(cluster_averages)):
                cluster_avg = cluster_averages[i]
                if is_similar_color(h, s, v, *cluster_avg, h_range=5, s_range=5, v_range=5):
                    clusters[i].append((x_coord, y_coord, hsv))
                    cluster_averages[i] = update_cluster_average(clusters[i])
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([(x_coord, y_coord, hsv)])
                cluster_averages.append((h, s, v))

    # Split connected components within each cluster
    new_clusters = []
    new_cluster_averages = []
    for i in range(len(clusters)):
        connected_components = split_connected_components(clusters[i])
        for component in connected_components:
            new_clusters.append(component)
            new_cluster_averages.append(update_cluster_average(component))

    return new_clusters, new_cluster_averages, width, height

# Plotting functions
def plot_cluster_points(points):
    plt.scatter(*zip(*points), s=10, color='blue', label="Cluster Points")
    plt.title("Cluster Points")
    plt.show()

def plot_convex_hull_boundary(points, width, height):
    hull = ConvexHull(points)
    plt.figure(figsize=(8, 6))
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

    plt.title("Convex Hull Boundary")
    plt.show()

# Main function
def main():
    image_path = "happymainimagemain.jpg"
    clusters, cluster_averages, width, height = load_and_cluster_image(image_path)

    for i in range(0,min(len(clusters),5)):
        for j in range(0,min(len(clusters[i]),5)):
            print(clusters[i][j],end="  ,  ")
        print("\n\n\n")


    # Calculate and plot  convex hull boundary for the first cluster as an example
    if clusters:
        cluster_points = [(x, y) for x, y, hsv in clusters[0]]

        # # Plot the raw cluster points for inspection
        # plot_cluster_points(cluster_points)

        # Generate and plot convex hull
        plot_convex_hull_boundary(np.array(cluster_points), width, height)




if __name__ == "__main__":
    main()
