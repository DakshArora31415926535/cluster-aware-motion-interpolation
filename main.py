import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_image(image_path, target_size=(170, 375)):
    """
    Load, convert to HSV, and resize an image using OpenCV.

    :param image_path: Path to the image file.
    :param target_size: Desired size for resizing the image (width, height).
    :return: Preprocessed HSV image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")
    image = cv2.resize(image, target_size)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def cluster_pixels_dbscan(hsv_image, eps=2, min_samples=5):
    flat_image = hsv_image.reshape((-1, 3))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(flat_image)
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {unique_labels}")
    print(f"Number of clusters (excluding noise): {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
    print(f"Number of noise points: {np.sum(labels == -1)}")
    clusters = labels.reshape(hsv_image.shape[:2])
    return clusters, labels

def extract_cluster_features(hsv_image, labels):
    height, width, _ = hsv_image.shape
    labels_2d = labels.reshape((height, width))
    clusters = np.unique(labels_2d)
    features = {}

    for cluster in clusters:
        if cluster == -1:
            continue

        mask = (labels_2d == cluster)
        cluster_pixels = hsv_image[mask]
        avg_hsv = np.mean(cluster_pixels, axis=0)
        cluster_size = len(cluster_pixels)
        features[cluster] = {
            'avg_hsv': avg_hsv,
            'size': cluster_size
        }
    return features

def visualize_cluster_boundaries(image, clusters):
    """
    Visualize the boundaries of clusters by drawing contours around them.

    :param image: Original image (in BGR format).
    :param clusters: 2D array of cluster labels.
    :return: Image with cluster boundaries drawn.
    """
    # Convert image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Iterate over each unique cluster label
    for cluster_label in np.unique(clusters):
        if cluster_label == -1:
            continue

        # Create a mask for the current cluster
        mask = np.zeros_like(gray)
        mask[clusters == cluster_label] = 255

        # Find contours of the cluster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return image

# Example usage
source_image_path = 'happymainimagemain.jpg'
target_image_path = 'sadmainimage.jpg'

source_image_hsv = preprocess_image(source_image_path)
source_image_bgr = cv2.imread(source_image_path)
target_image_hsv = preprocess_image(target_image_path)

source_clusters, source_labels = cluster_pixels_dbscan(source_image_hsv)

# Visualize cluster boundaries on the source image
visualized_image = visualize_cluster_boundaries(source_image_bgr, source_clusters)

# Display the image with cluster boundaries
cv2.imshow("Cluster Boundaries", visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
