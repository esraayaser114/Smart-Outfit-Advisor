import cv2
import numpy as np
from sklearn.cluster import KMeans


# =========================
# Fashion Color Palette
# =========================

FASHION_COLORS = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Dark Gray": (64, 64, 64),
    "Gray": (128, 128, 128),
    "Light Gray": (211, 211, 211),

    "Beige": (245, 245, 220),
    "Cream": (255, 253, 208),
    "Tan": (210, 180, 140),
    "Brown": (101, 67, 33),

    "Red": (220, 20, 60),
    "Burgundy": (128, 0, 32),
    "Pink": (255, 105, 180),
    "Light Pink": (255, 182, 193),
    "Coral": (255, 127, 80),

    "Orange": (255, 165, 0),
    "Yellow": (255, 215, 0),
    "Mustard": (204, 153, 0),

    "Green": (34, 139, 34),
    "Dark Green": (0, 100, 0),
    "Olive": (128, 128, 0),

    "Light Blue": (173, 216, 230),
    "Blue": (70, 130, 180),
    "Navy": (25, 25, 112),
    "Turquoise": (64, 224, 208),

    "Purple": (128, 0, 128),
    "Lavender": (216, 191, 216),
}


# =========================
# Neutral Colors
# =========================

NEUTRAL_COLORS = {
    "Black",
    "White",
    "Gray",
    "Dark Gray",
    "Light Gray",
    "Beige",
    "Cream",
}


# =========================
# Extract Dominant Colors
# =========================

def extract_top_colors(image_path, k=3):
    """
    Extract the dominant colors from an image using K-Means clustering.
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    pixels = image_rgb.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    # Get cluster centers as RGB colors
    colors = kmeans.cluster_centers_.astype(int)

    # Count pixels in each cluster
    counts = np.bincount(kmeans.labels_)

    # Convert pixel counts to percentages
    percentages = (
        counts / counts.sum()
    ) * 100

    # Sort colors by dominance
    order = np.argsort(percentages)[::-1]

    return (
        colors[order],
        percentages[order]
    )


# =========================
# RGB → Fashion Color Name
# =========================

def get_fashion_color_name(rgb):
    """
    Map an RGB color to the nearest fashion color.
    """

    r, g, b = map(int, rgb)

    closest_name = None
    min_distance = float("inf")

    for color_name, color_rgb in FASHION_COLORS.items():

        cr, cg, cb = color_rgb

        # Calculate Euclidean RGB distance
        distance = np.sqrt(
            (r - cr) ** 2 +
            (g - cg) ** 2 +
            (b - cb) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_name = color_name

    return closest_name


# =========================
# RGB → HSV
# =========================

def rgb_to_hsv(rgb):
    """
    Convert an RGB color to HSV.
    Hue is returned in degrees.
    Saturation and Value are returned as percentages.
    """

    rgb_array = np.uint8([[rgb]])

    hsv = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV
    )[0][0]

    # Convert Hue from 0-179 to 0-360
    h = int(hsv[0]) * 2

    # Convert Saturation and Value to percentages
    s = int(hsv[1]) / 255 * 100
    v = int(hsv[2]) / 255 * 100

    return h, s, v


# =========================
# Hue Difference
# =========================

def hue_difference(h1, h2):
    """
    Calculate the shortest distance between two hues
    on the color wheel.
    """

    difference = abs(h1 - h2)

    return min(
        difference,
        360 - difference
    )


# =========================
# Color Harmony Analysis
# =========================

def analyze_color_harmony(
    chromatic_items,
    neutral_items
):
    """
    Analyze the color harmony of an outfit.
    """

    number_of_colors = len(chromatic_items)

    # All colors are neutral
    if number_of_colors == 0:
        return "Neutral Harmony"

    # One chromatic color
    if number_of_colors == 1:

        if neutral_items:
            return "Neutral-Based Harmony"

        return "Single Color"

    # Extract Hue values
    hues = [
        item["hsv"][0]
        for item in chromatic_items
    ]

    # Two chromatic colors
    if number_of_colors == 2:

        difference = hue_difference(
            hues[0],
            hues[1]
        )

        if difference <= 15:
            return "Monochromatic"

        elif difference <= 45:
            return "Analogous"

        elif 150 <= difference <= 210:
            return "Complementary"

        return "No Standard Harmony"

    # Three or more chromatic colors
    h1, h2, h3 = hues[:3]

    differences = [
        hue_difference(h1, h2),
        hue_difference(h1, h3),
        hue_difference(h2, h3)
    ]

    # Monochromatic harmony
    if max(differences) <= 15:
        return "Monochromatic"

    # Analogous harmony
    if max(differences) <= 60:
        return "Analogous"

    # Triadic harmony
    if all(
        90 <= difference <= 150
        for difference in differences
    ):
        return "Triadic"

    # Complementary-based harmony
    if any(
        150 <= difference <= 210
        for difference in differences
    ):
        return "Complementary-Based Harmony"

    return "No Standard Harmony"


# =========================
# Color Harmony Score
# =========================

def get_color_score(harmony):
    """
    Return a color harmony score from 0 to 10.
    """

    scores = {
        "Neutral Harmony": 9.0,
        "Neutral-Based Harmony": 9.0,
        "Monochromatic": 9.0,
        "Analogous": 8.5,
        "Complementary": 9.0,
        "Triadic": 8.5,
        "Complementary-Based Harmony": 8.0,
        "Single Color": 7.5,
        "No Standard Harmony": 4.5
    }

    return scores.get(
        harmony,
        5.0
    )