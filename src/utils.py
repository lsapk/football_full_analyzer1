import math
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image, box, k=3):
    """
    Finds the dominant color in a cropped region of an image using K-Means clustering.
    The crop focuses on the player's torso to better identify jersey color.

    Args:
        image (np.array): The full frame.
        box (list or tuple): The bounding box [x1, y1, x2, y2].
        k (int): The number of clusters for K-Means.

    Returns:
        tuple: The dominant color in BGR format, or None if the box is invalid.
    """
    x1, y1, x2, y2 = map(int, box)
    if x1 >= x2 or y1 >= y2:
        return None

    # Crop to the torso area to avoid head/legs
    height = y2 - y1
    torso_y1 = y1 + int(height * 0.15)
    torso_y2 = y1 + int(height * 0.50)
    torso_x1 = x1 + int((x2-x1)*0.1)
    torso_x2 = x2 - int((x2-x1)*0.1)

    if torso_y1 >= torso_y2 or torso_x1 >= torso_x2:
        # Fallback to the full box if torso crop is too small
        torso_y1, torso_y2, torso_x1, torso_x2 = y1, y2, x1, x2
        if torso_y1 >= torso_y2 or torso_x1 >= torso_x2:
             return None


    player_crop = image[torso_y1:torso_y2, torso_x1:torso_x2]

    if player_crop.size == 0:
        return None

    # Convert BGR to HSV for more robust color clustering
    hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

    # Reshape the image to be a list of pixels
    pixels = hsv_crop.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Perform K-Means clustering in HSV space
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Find the most frequent cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster_idx = unique[np.argmax(counts)]
    dominant_hsv_color = kmeans.cluster_centers_[dominant_cluster_idx]

    # Convert the dominant HSV color back to BGR for display
    dominant_bgr_color = cv2.cvtColor(np.uint8([[dominant_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

    return tuple(map(int, dominant_bgr_color))

def box_center(box):
    """
    Calculates the center coordinates of a bounding box.
    Args:
        box (list or tuple): A list of 4 coordinates [x1, y1, x2, y2].
    Returns:
        A tuple (x, y) representing the center of the box.
    """
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    return (x, y)

def pixel_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points in pixels.
    Args:
        p1 (tuple): The first point (x1, y1).
        p2 (tuple): The second point (x2, y2).
    Returns:
        The distance in pixels.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def speed_kmh(pixels, dt_seconds, pixels_to_meters):
    """
    Converts a speed from pixels per frame to km/h.
    Args:
        pixels (float): The distance traveled in pixels.
        dt_seconds (float): The time elapsed in seconds.
        pixels_to_meters (float): The conversion factor from pixels to meters.
    Returns:
        The speed in km/h.
    """
    if dt_seconds <= 0:
        return 0.0
    meters = pixels * pixels_to_meters
    m_per_s = meters / dt_seconds
    return m_per_s * 3.6