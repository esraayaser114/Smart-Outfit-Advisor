import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Outfit Advisor",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# FASHION COLOR PALETTE
# ============================================================

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


# ============================================================
# NEUTRAL COLORS
# ============================================================

NEUTRAL_COLORS = {
    "Black",
    "White",
    "Gray",
    "Dark Gray",
    "Light Gray",
    "Beige",
    "Cream",
}


# ============================================================
# LOAD YOLO MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "best_yolo26l.pt"


@st.cache_resource
def load_model():
    """
    Load the YOLO model only once.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model was not found:\n{MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# ============================================================
# DOMINANT COLOR USING K-MEANS
# ============================================================

def get_dominant_color(image, k=3):
    """
    Extract the dominant RGB color from an image crop
    using K-Means clustering.
    """

    if image is None:
        return None

    if image.size == 0:
        return None

    # OpenCV image is BGR.
    # Convert it to RGB.
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pixels = image_rgb.reshape(-1, 3)

    # If there are fewer pixels than clusters
    if len(pixels) < k:
        return np.mean(
            pixels,
            axis=0
        ).astype(int)

    # Use at most 10,000 pixels
    # to make K-Means faster.
    if len(pixels) > 10000:

        rng = np.random.default_rng(42)

        indexes = rng.choice(
            len(pixels),
            size=10000,
            replace=False
        )

        pixels = pixels[indexes]

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    counts = np.bincount(
        kmeans.labels_
    )

    dominant_index = np.argmax(
        counts
    )

    dominant_color = (
        kmeans.cluster_centers_[dominant_index]
    )

    return dominant_color.astype(int)


# ============================================================
# RGB TO FASHION COLOR NAME
# ============================================================

def get_fashion_color_name(rgb):
    """
    Find the closest fashion color using RGB distance.
    """

    r, g, b = map(
        int,
        rgb
    )

    closest_name = None
    minimum_distance = float("inf")

    for name, color in FASHION_COLORS.items():

        cr, cg, cb = color

        distance = np.sqrt(
            (r - cr) ** 2
            + (g - cg) ** 2
            + (b - cb) ** 2
        )

        if distance < minimum_distance:

            minimum_distance = distance
            closest_name = name

    return closest_name


# ============================================================
# RGB TO HSV
# ============================================================

def rgb_to_hsv(rgb):
    """
    Convert RGB to HSV.

    Returns:
        H = degrees 0-360
        S = percentage 0-100
        V = percentage 0-100
    """

    rgb_array = np.uint8([[rgb]])

    hsv = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV
    )[0][0]

    hue = int(hsv[0]) * 2

    saturation = (
        int(hsv[1]) / 255
    ) * 100

    value = (
        int(hsv[2]) / 255
    ) * 100

    return (
        hue,
        saturation,
        value
    )


# ============================================================
# HUE DIFFERENCE
# ============================================================

def hue_difference(h1, h2):
    """
    Calculate the shortest distance between two hues.
    """

    difference = abs(
        h1 - h2
    )

    return min(
        difference,
        360 - difference
    )


# ============================================================
# COLOR HARMONY
# ============================================================

def analyze_color_harmony(
    chromatic_items,
    neutral_items
):
    """
    Analyze color harmony for detected clothing items.
    """

    number_of_colors = len(
        chromatic_items
    )

    # --------------------------------------------------------
    # No chromatic colors
    # --------------------------------------------------------

    if number_of_colors == 0:

        return "Neutral Harmony"

    # --------------------------------------------------------
    # One chromatic color
    # --------------------------------------------------------

    if number_of_colors == 1:

        if len(neutral_items) > 0:
            return "Neutral-Based Harmony"

        return "Single Color"

    # --------------------------------------------------------
    # Extract hues
    # --------------------------------------------------------

    hues = [
        item["hsv"][0]
        for item in chromatic_items
    ]

    # --------------------------------------------------------
    # Two chromatic colors
    # --------------------------------------------------------

    if number_of_colors == 2:

        difference = hue_difference(
            hues[0],
            hues[1]
        )

        if difference <= 15:
            return "Monochromatic"

        if difference <= 45:
            return "Analogous"

        if difference >= 150:
            return "Complementary"

        return "No Standard Harmony"

    # --------------------------------------------------------
    # Three or more colors
    # --------------------------------------------------------

    h1 = hues[0]
    h2 = hues[1]
    h3 = hues[2]

    differences = [
        hue_difference(h1, h2),
        hue_difference(h1, h3),
        hue_difference(h2, h3)
    ]

    # Monochromatic
    if max(differences) <= 15:

        return "Monochromatic"

    # Analogous
    if max(differences) <= 60:

        return "Analogous"

    # Triadic
    if all(
        90 <= difference <= 150
        for difference in differences
    ):

        return "Triadic"

    # Complementary
    if any(
        difference >= 150
        for difference in differences
    ):

        return "Complementary-Based Harmony"

    return "No Standard Harmony"


# ============================================================
# COLOR SCORE
# ============================================================

def get_color_score(harmony):
    """
    Convert color harmony into a score from 0 to 10.
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
        "No Standard Harmony": 4.5,
    }

    return scores.get(
        harmony,
        5.0
    )


# ============================================================
# CROP CLOTHING ITEM
# ============================================================

def crop_item(image, bbox):
    """
    Crop one detected clothing item.
    """

    x1, y1, x2, y2 = bbox

    height, width = image.shape[:2]

    x1 = max(
        0,
        min(x1, width)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if x2 <= x1 or y2 <= y1:

        return None

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_items(model, image):
    """
    Detect clothing items using YOLO.
    """

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    detections = []

    # Annotated image
    annotated_image = result.plot()

    if result.boxes is None:

        return (
            detections,
            annotated_image
        )

    names = result.names

    for box in result.boxes:

        confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        label = names[class_id]

        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
            }
        )

    return (
        detections,
        annotated_image
    )


# ============================================================
# SUITABILITY
# ============================================================

def calculate_suitability(
    occasion,
    time,
    temperature,
    weather
):
    """
    Calculate outfit suitability based on context.
    """

    score = 10.0

    reasons = []

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if temperature >= 32:

        score -= 1.5

        reasons.append(
            "The temperature is high, so lightweight clothing is recommended."
        )

    elif temperature >= 27:

        score -= 0.5

        reasons.append(
            "The weather is warm, so avoid heavy layers."
        )

    elif temperature <= 12:

        score -= 1.0

        reasons.append(
            "The weather is cold, so warmer layers are recommended."
        )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if weather == "Rainy":

        score -= 1.5

        reasons.append(
            "Rainy weather: waterproof clothing is recommended."
        )

    elif weather == "Sunny":

        reasons.append(
            "Sunny weather is generally suitable."
        )

    elif weather == "Cold":

        score -= 0.5

        reasons.append(
            "Cold conditions: consider adding a warm layer."
        )

    elif weather == "Hot":

        score -= 1.0

        reasons.append(
            "Hot conditions: lightweight fabrics are recommended."
        )

    # --------------------------------------------------------
    # Occasion
    # --------------------------------------------------------

    if occasion == "University":

        reasons.append(
            "University is suitable for comfortable casual outfits."
        )

    elif occasion == "Work":

        score -= 1.0

        reasons.append(
            "Work usually requires a more polished appearance."
        )

    elif occasion == "Party":

        score += 0.5

        reasons.append(
            "Party settings allow more expressive styling."
        )

    elif occasion == "Dinner":

        score += 0.2

        reasons.append(
            "Dinner outfits generally benefit from a polished appearance."
        )

    elif occasion == "Casual":

        reasons.append(
            "Casual settings allow flexible outfit choices."
        )

    elif occasion == "Sport":

        score -= 1.5

        reasons.append(
            "Sport activities require functional sportswear."
        )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    if time == "Morning":

        reasons.append(
            "Morning outfits are usually best when practical and comfortable."
        )

    elif time == "Afternoon":

        reasons.append(
            "Afternoon settings are generally flexible."
        )

    elif time == "Evening":

        reasons.append(
            "Evening outfits can be slightly more polished."
        )

    # Keep score between 0 and 10
    score = max(
        0.0,
        min(10.0, score)
    )

    return (
        round(score, 1),
        reasons
    )


# ============================================================
# FINAL RATING
# ============================================================

def calculate_final_rating(
    detection_score,
    color_score,
    suitability_score
):
    """
    Calculate the final outfit score.
    """

    final_score = (
        detection_score * 0.30
        + color_score * 0.40
        + suitability_score * 0.30
    )

    return round(
        final_score,
        2
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("👗 Smart Outfit Advisor")

st.write(
    "Analyze your outfit using clothing detection, "
    "color harmony and context suitability."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🧠 Outfit Context"
)

occasion = st.sidebar.selectbox(
    "📍 Occasion / Place",
    [
        "University",
        "Work",
        "Party",
        "Dinner",
        "Casual",
        "Sport",
    ]
)

time = st.sidebar.selectbox(
    "🕐 Time",
    [
        "Morning",
        "Afternoon",
        "Evening",
    ]
)

temperature = st.sidebar.number_input(
    "🌡️ Temperature (°C)",
    min_value=-20,
    max_value=50,
    value=25
)

weather = st.sidebar.selectbox(
    "🌤️ Weather",
    [
        "Sunny",
        "Rainy",
        "Cold",
        "Hot",
    ]
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📸 Upload Outfit Image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload an outfit image to start the analysis."
    )

else:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "❌ Could not read the uploaded image."
        )

        st.stop()

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    st.subheader(
        "📷 Uploaded Outfit"
    )

    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    analyze_button = st.button(
        "✨ Analyze Outfit",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        # ====================================================
        # LOAD MODEL
        # ====================================================

        try:

            with st.spinner(
                "🤖 Loading YOLO model..."
            ):

                model = load_model()

        except Exception as error:

            st.error(
                "❌ Could not load the YOLO model."
            )

            st.exception(error)

            st.stop()

        # ====================================================
        # DETECTION
        # ====================================================

        try:

            with st.spinner(
                "🔍 Detecting clothing items..."
            ):

                detections, annotated_image = detect_items(
                    model,
                    image
                )

        except Exception as error:

            st.error(
                "❌ Clothing detection failed."
            )

            st.exception(error)

            st.stop()

        # ====================================================
        # DETECTION RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "👕 Detected Clothing Items"
        )

        if len(detections) == 0:

            st.warning(
                "⚠️ No clothing items were detected."
            )

            st.stop()

        # Display YOLO result
        st.image(
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            ),
            caption="YOLO Detection",
            use_container_width=True
        )

        # ====================================================
        # COLOR ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "🎨 Color Analysis"
        )

        analyzed_items = []

        chromatic_items = []

        neutral_items = []

        color_names = []

        # ----------------------------------------------------
        # Analyze each detected item
        # ----------------------------------------------------

        for item in detections:

            crop = crop_item(
                image,
                item["bbox"]
            )

            if crop is None:
                continue

            dominant_rgb = get_dominant_color(
                crop,
                k=3
            )

            if dominant_rgb is None:
                continue

            color_name = get_fashion_color_name(
                dominant_rgb
            )

            hsv = rgb_to_hsv(
                dominant_rgb
            )

            analyzed_item = {
                "label": item["label"],
                "confidence": item["confidence"],
                "rgb": dominant_rgb.tolist(),
                "color_name": color_name,
                "hsv": hsv,
                "crop": crop,
            }

            analyzed_items.append(
                analyzed_item
            )

            color_names.append(
                color_name
            )

            if color_name in NEUTRAL_COLORS:

                neutral_items.append(
                    analyzed_item
                )

            else:

                chromatic_items.append(
                    analyzed_item
                )

        # ====================================================
        # DISPLAY ITEM COLORS
        # ====================================================

        if len(analyzed_items) > 0:

            number_of_columns = min(
                3,
                len(analyzed_items)
            )

            columns = st.columns(
                number_of_columns
            )

            for index, item in enumerate(
                analyzed_items
            ):

                with columns[
                    index % number_of_columns
                ]:

                    st.image(
                        cv2.cvtColor(
                            item["crop"],
                            cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True
                    )

                    st.write(
                        f"**{item['label']}**"
                    )

                    st.write(
                        f"Color: **{item['color_name']}**"
                    )

                    r, g, b = item["rgb"]

                    st.write(
                        f"RGB: ({r}, {g}, {b})"
                    )

                    h, s, v = item["hsv"]

                    st.write(
                        f"HSV: ({h}°, {s:.1f}%, {v:.1f}%)"
                    )

                    st.write(
                        f"Confidence: {item['confidence']:.2f}"
                    )

        # ====================================================
        # COLOR HARMONY
        # ====================================================

        harmony = analyze_color_harmony(
            chromatic_items,
            neutral_items
        )

        color_score = get_color_score(
            harmony
        )

        st.divider()

        color_col1, color_col2 = st.columns(2)

        with color_col1:

            st.metric(
                "🎨 Color Harmony",
                harmony
            )

        with color_col2:

            st.metric(
                "🎨 Color Score",
                f"{color_score}/10"
            )

        # ====================================================
        # SUITABILITY
        # ====================================================

        suitability_score, reasons = calculate_suitability(
            occasion,
            time,
            temperature,
            weather
        )

        st.divider()

        st.subheader(
            "🧠 Context & Suitability"
        )

        context_col1, context_col2 = st.columns(2)

        with context_col1:

            st.write(
                f"📍 **Occasion:** {occasion}"
            )

            st.write(
                f"🕐 **Time:** {time}"
            )

        with context_col2:

            st.write(
                f"🌡️ **Temperature:** {temperature}°C"
            )

            st.write(
                f"🌤️ **Weather:** {weather}"
            )

        st.metric(
            "🧠 Suitability Score",
            f"{suitability_score}/10"
        )

        if len(reasons) > 0:

            st.write(
                "**Analysis:**"
            )

            for reason in reasons:

                st.write(
                    f"• {reason}"
                )

        # ====================================================
        # DETECTION SCORE
        # ====================================================

        confidence_values = [
            item["confidence"]
            for item in detections
        ]

        average_confidence = np.mean(
            confidence_values
        )

        detection_score = round(
            average_confidence * 10,
            1
        )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        final_score = calculate_final_rating(
            detection_score,
            color_score,
            suitability_score
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "⭐ Final Outfit Rating"
        )

        st.metric(
            "FINAL SCORE",
            f"{final_score}/10"
        )

        # ----------------------------------------------------
        # Rating message
        # ----------------------------------------------------

        if final_score >= 9:

            st.success(
                "🔥 Excellent outfit! The colors and context work very well together."
            )

        elif final_score >= 8:

            st.success(
                "✨ Very good outfit with strong overall coordination."
            )

        elif final_score >= 7:

            st.info(
                "👍 Good outfit with some room for improvement."
            )

        elif final_score >= 5:

            st.warning(
                "⚠️ Average outfit. Some aspects could be improved."
            )

        else:

            st.error(
                "❌ The outfit may need significant improvements."
            )

        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        st.subheader(
            "📊 Score Breakdown"
        )

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:

            st.metric(
                "Detection",
                f"{detection_score}/10"
            )

        with score_col2:

            st.metric(
                "Color",
                f"{color_score}/10"
            )

        with score_col3:

            st.metric(
                "Suitability",
                f"{suitability_score}/10"
            )

        st.caption(
            "Final Score = 30% Detection + 40% Color + 30% Suitability"
        )