import math

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