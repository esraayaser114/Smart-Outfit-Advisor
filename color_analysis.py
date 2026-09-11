import cv2
import numpy as np
from sklearn.cluster import KMeans


# ==========================================
# Fashion Color Palette K-means
# ==========================================

# Reference RGB values for common fashion colors
FASHION_COLORS = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128),
    "Red": (220, 50, 50),
    "Orange": (240, 140, 40),
    "Yellow": (240, 210, 50),
    "Green": (60, 160, 80),
    "Blue": (60, 100, 200),
    "Navy": (30, 50, 120),
    "Purple": (140, 80, 170),
    "Pink": (230, 120, 170),
    "Brown": (130, 80, 40),
    "Beige": (220, 200, 160),
}


# ==========================================
# RGB → HSV
# Hue = 0–360
# Saturation = 0–1
# Value = 0–1
# ==========================================

def rgb_to_hsv(rgb):
    """
    Convert an RGB color into HSV color space.

    Hue represents the color angle from 0 to 360 degrees.
    Saturation represents color intensity from 0 to 1.
    Value represents brightness from 0 to 1.
    """

    # Convert the RGB values into a NumPy image array
    rgb_array = np.uint8([
        [[
            int(rgb[0]),
            int(rgb[1]),
            int(rgb[2])
        ]]
    ])

    # Convert RGB image to HSV using OpenCV
    hsv = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV
    )[0][0]

    # OpenCV stores Hue between 0 and 180,
    # so multiply by 2 to get the range 0–360
    h = float(hsv[0]) * 2

    # Convert Saturation from 0–255 to 0–1
    s = float(hsv[1]) / 255

    # Convert Value from 0–255 to 0–1
    v = float(hsv[2]) / 255

    return h, s, v


# ==========================================
# Get Nearest Fashion Color
# ==========================================

def get_fashion_color_name(rgb):
    """
    Find the closest predefined fashion color
    based on Euclidean distance in RGB space.
    """

    # Convert the input RGB color into a NumPy array
    rgb = np.array(rgb, dtype=float)

    best_name = None
    best_distance = float("inf")

    # Compare the input color with every reference color
    for name, reference_rgb in FASHION_COLORS.items():

        # Convert the reference RGB color to a NumPy array
        reference = np.array(
            reference_rgb,
            dtype=float
        )

        # Calculate the Euclidean distance between the colors
        distance = np.linalg.norm(
            rgb - reference
        )

        # Keep the color with the smallest distance
        if distance < best_distance:

            best_distance = distance
            best_name = name

    return best_name


# ==========================================
# Dominant Color
# ==========================================

def extract_dominant_color(
    image_bgr,
    k=4
):
    """
    Extract the dominant color from a clothing image
    using K-Means clustering.
    """

    # Return None if the image does not exist
    if image_bgr is None:
        return None

    # Return None if the image is empty
    if image_bgr.size == 0:
        return None

    # Get the image dimensions
    h, w = image_bgr.shape[:2]

    # Define a 10% margin to reduce the effect
    # of borders and background pixels
    margin_y = max(1, int(h * 0.10))
    margin_x = max(1, int(w * 0.10))

    # Remove the outer borders of the cropped clothing image
    image_bgr = image_bgr[
        margin_y:h-margin_y,
        margin_x:w-margin_x
    ]

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Reshape the image into a list of RGB pixels
    pixels = image_rgb.reshape(
        -1,
        3
    )

    # Limit the number of pixels used by K-Means
    # to improve performance
    if len(pixels) > 10000:

        # Randomly select 10,000 pixels
        indices = np.random.choice(
            len(pixels),
            10000,
            replace=False
        )

        pixels = pixels[indices]

    # Make sure the number of clusters
    # does not exceed the number of available pixels
    k = min(k, len(pixels))

    # Return None if there are no valid pixels
    if k < 1:
        return None

    # Create the K-Means clustering model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Assign each pixel to the closest color cluster
    labels = kmeans.fit_predict(pixels)

    # Get the RGB center of each cluster
    centers = kmeans.cluster_centers_

    # Count how many pixels belong to each cluster
    counts = np.bincount(labels)

    # Find the cluster containing the largest number of pixels
    dominant_index = np.argmax(counts)

    # Get the RGB value of the dominant cluster
    dominant_rgb = np.round(
        centers[dominant_index]
    ).astype(int)

    # Calculate the percentage of pixels
    # represented by the dominant color
    percentage = (
        counts[dominant_index]
        / len(labels)
        * 100
    )

    # Convert the RGB NumPy array into a Python tuple
    dominant_rgb = tuple(
        dominant_rgb.tolist()
    )

    # Convert the dominant RGB color to HSV
    hsv = rgb_to_hsv(
        dominant_rgb
    )

    # Find the nearest predefined fashion color name
    color_name = get_fashion_color_name(
        dominant_rgb
    )

    # Return all extracted color information
    return {
        "rgb": dominant_rgb,
        "hsv": hsv,
        "color_name": color_name,
        "percentage": round(
            float(percentage),
            2
        )
    }


# ==========================================
# Analyze Outfit Colors
# ==========================================

def analyze_colors(cropped_items):
    """
    Analyze the dominant color of each detected clothing item.
    """

    # Create a list to store the analyzed items
    analyzed_items = []

    # Process every cropped clothing item
    for item in cropped_items:

        # Extract the dominant color from the clothing crop
        color_info = extract_dominant_color(
            item["crop"]
        )

        # Skip the item if color extraction failed
        if color_info is None:
            continue

        # Combine the original detection information
        # with the extracted color information
        analyzed_items.append({
            **item,
            **color_info
        })

    return analyzed_items


# ==========================================
# Hue Difference
# ==========================================

def hue_difference(h1, h2):
    """
    Calculate the shortest distance between two hue values.

    Since hue is circular, the difference between
    350° and 10° is 20°, not 340°.
    """

    # Calculate the direct absolute difference
    diff = abs(h1 - h2)

    # Return the shortest circular distance
    return min(
        diff,
        360 - diff
    )


# ==========================================
# Color Harmony
# ==========================================

def analyze_color_harmony(outfit_items):
    """
    Analyze the color harmony of the detected outfit.

    The function classifies the outfit into categories such as:
    Neutral, Monochromatic, Analogous, Complementary, or Triadic.
    """

    # Return this result when no clothing colors are available
    if not outfit_items:
        return "No Colors"

    # Separate neutral and colorful clothing items
    neutral_items = []
    chromatic_items = []

    # Analyze each clothing item's HSV values
    for item in outfit_items:

        h, s, v = item["hsv"]

        # Low saturation or very low brightness
        # indicates a neutral color
        if s < 0.20 or v < 0.15:
            neutral_items.append(item)

        else:
            chromatic_items.append(item)

    # Handle outfits containing only neutral colors
    if len(chromatic_items) == 0:

        if len(neutral_items) >= 2:
            return "Neutral"

        return "Single Color"

    # Handle outfits containing only one colorful item
    if len(chromatic_items) == 1:

        if neutral_items:
            return "Neutral-Based Harmony"

        return "Single Color"

    # Extract the hue values of all colorful items
    hues = [
        item["hsv"][0]
        for item in chromatic_items
    ]

    # ==========================================
    # Two-Color Harmony
    # ==========================================

    if len(hues) == 2:

        # Calculate the hue difference between the two colors
        diff = hue_difference(
            hues[0],
            hues[1]
        )

        # Very small hue difference indicates monochromatic colors
        if diff <= 15:
            return "Monochromatic"

        # Nearby hues indicate analogous harmony
        elif diff <= 60:
            return "Analogous"

        # Opposite hues indicate complementary harmony
        elif diff >= 150:
            return "Complementary"

        # No standard harmony pattern was detected
        return "No Standard Harmony"

    # ==========================================
    # Three or More Colors
    # ==========================================

    # Calculate the hue difference between every pair of colors
    pairwise_diffs = [
        hue_difference(hues[i], hues[j])
        for i in range(len(hues))
        for j in range(i + 1, len(hues))
    ]

    # Check for monochromatic harmony
    if max(pairwise_diffs) <= 15:
        return "Monochromatic"

    # Check for analogous harmony
    if max(pairwise_diffs) <= 60:
        return "Analogous"

    # ==========================================
    # Triadic Harmony
    # ==========================================

    # Sort the hue values and use the first three
    # for the triadic harmony check
    sorted_hues = sorted(hues)[:3]

    # Calculate the angular distances between the three hues
    d1 = sorted_hues[1] - sorted_hues[0]
    d2 = sorted_hues[2] - sorted_hues[1]
    d3 = 360 - (
        sorted_hues[2] - sorted_hues[0]
    )

    # A triadic color scheme has approximately
    # 120 degrees between each color
    if all(
        abs(d - 120) <= 20
        for d in [d1, d2, d3]
    ):
        return "Triadic"

    # Check if at least one pair of colors is complementary
    if any(
        diff >= 150
        for diff in pairwise_diffs
    ):
        return "Complementary-Based Harmony"

    # Return this when no standard harmony pattern is detected
    return "No Standard Harmony"


# ==========================================
# Color Score
# ==========================================

# Scores assigned to each color harmony category
HARMONY_SCORES = {

    "Neutral": 9.0,

    "Neutral-Based Harmony": 9.0,

    "Single Color": 7.5,

    "Monochromatic": 9.0,

    "Analogous": 8.5,

    "Complementary": 9.0,

    "Triadic": 8.5,

    "Complementary-Based Harmony": 8.0,

    "No Standard Harmony": 5.0,

    "No Colors": 0.0
}


def get_color_score(harmony):
    """
    Return the numerical score assigned to a color harmony type.
    """

    # Get the score from the predefined dictionary.
    # Return 5.0 if the harmony type is not found.
    return HARMONY_SCORES.get(
        harmony,
        5.0
    )