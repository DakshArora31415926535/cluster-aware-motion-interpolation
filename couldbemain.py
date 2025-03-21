from PIL import Image, ImageDraw
import multiprocessing as mp
import colorsys
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ------------------------------- #
#         HELPER FUNCTIONS        #
# ------------------------------- #

def rgb_to_hsv(r, g, b):
    """Converts RGB values to HSV"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100  # Scale HSV values


def is_similar_color(h1, s1, v1, h2, s2, v2, h_range=5, s_range=5, v_range=5):
    """Checks if two HSV colors are within a defined similarity threshold"""
    return abs(h1 - h2) <= h_range and abs(s1 - s2) <= s_range and abs(v1 - v2) <= v_range


def process_pixel(args):
    """Extracts HSV values for a given pixel"""
    x, y, rgb_image = args
    r, g, b = rgb_image.getpixel((x, y))
    return x, y, rgb_to_hsv(r, g, b)


def cluster_image(image_path):
    """Clusters the image using color similarity & connected components"""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    pool = mp.Pool(mp.cpu_count())
    hsv_values = pool.map(process_pixel, [(x, y, image) for x in range(width) for y in range(height)])
    pool.close()
    pool.join()

    clusters = []
    cluster_averages = []

    for x, y, hsv in hsv_values:
        found_cluster = False
        for i in range(len(cluster_averages)):
            if is_similar_color(hsv[0], hsv[1], hsv[2], *cluster_averages[i]):
                clusters[i].append((x, y, hsv))
                cluster_averages[i] = np.mean([hsv for _, _, hsv in clusters[i]], axis=0)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([(x, y, hsv)])
            cluster_averages.append(hsv)

    return clusters, cluster_averages, width, height


def convex_hull(points):
    """Computes convex hull for a given set of cluster points"""
    if len(points) < 3:
        return points  # Not enough points for a hull
    hull = ConvexHull(points)
    return [points[i] for i in hull.vertices]


def bezier_curve(t, p0, p1, p2):
    """Quadratic Bezier curve interpolation"""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


# ------------------------------- #
#     FRAME GENERATION LOGIC      #
# ------------------------------- #

def map_clusters_between_frames(clusters_1, clusters_2):
    """
    Establishes correspondence between clusters in two consecutive frames
    """
    mappings = {}
    for i, cluster_1 in enumerate(clusters_1):
        min_distance = float("inf")
        best_match = None
        centroid_1 = np.mean([(x, y) for x, y, _ in cluster_1], axis=0)

        for j, cluster_2 in enumerate(clusters_2):
            centroid_2 = np.mean([(x, y) for x, y, _ in cluster_2], axis=0)
            dist = np.linalg.norm(centroid_1 - centroid_2)
            if dist < min_distance:
                min_distance = dist
                best_match = j

        mappings[i] = best_match
    return mappings


def generate_intermediate_frames(image1_path, image2_path, num_frames=5):
    """
    Generates interpolated frames between two images based on cluster movement
    """

    print("jjhjhkgghg")
    # Load and cluster both images
    clusters_1, cluster_avgs_1, width, height = cluster_image(image1_path)
    print("nbnyudydtf")
    clusters_2, cluster_avgs_2, _, _ = cluster_image(image2_path)

    # Establish cluster correspondences
    mappings = map_clusters_between_frames(clusters_1, clusters_2)

    # Generate interpolated frames
    intermediate_images = []
    for t in np.linspace(0, 1, num_frames + 2)[1:-1]:  # Skip first & last (original frames)
        frame = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(frame)

        for i, cluster in enumerate(clusters_1):
            if i not in mappings or mappings[i] is None:
                continue  # No matching cluster found

            cluster_2 = clusters_2[mappings[i]]
            points_1 = [(x, y) for x, y, _ in cluster]
            points_2 = [(x, y) for x, y, _ in cluster_2]

            if len(points_1) < 3 or len(points_2) < 3:
                continue  # Skip small clusters

            hull_1 = convex_hull(points_1)
            hull_2 = convex_hull(points_2)

            # Interpolate cluster positions
            interpolated_hull = [(int(bezier_curve(t, p1[0], (p1[0] + p2[0]) / 2, p2[0])),
                                  int(bezier_curve(t, p1[1], (p1[1] + p2[1]) / 2, p2[1])))
                                 for p1, p2 in zip(hull_1, hull_2)]

            avg_color = np.mean([hsv for _, _, hsv in cluster], axis=0)
            color_rgb = tuple(
                int(c) for c in colorsys.hsv_to_rgb(avg_color[0] / 360, avg_color[1] / 100, avg_color[2] / 100))

            draw.polygon(interpolated_hull, fill=color_rgb)

        intermediate_images.append(frame)

    # Save frames
    for idx, frame in enumerate(intermediate_images):
        frame.save(f"interpolated_frame_{idx + 1}.png")


# ------------------------------- #
#           EXECUTION             #
# ------------------------------- #

if __name__ == "__main__":
    print("hello")
    img1 = "happymainimagemain.jpg"
    img2 = "sadmainimage.jpg"
    print("hgjjjmkfkj")
    generate_intermediate_frames(img1, img2, num_frames=1)
    print("Intermediate frames generated successfully!")
