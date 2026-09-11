import cv2

from clothing_detection import (
    detect_clothing,
    crop_detected_items
)

from color_analysis import (
    analyze_colors,
    analyze_color_harmony,
    get_color_score
)

from suitability import (
    evaluate_outfit_suitability
)


# ==========================================
# Complete Outfit Analysis
# ==========================================

def analyze_outfit(
    image,
    occasion,
    time_of_day,
    weather,
    temperature
):
    """
    Perform a complete outfit analysis.

    The analysis includes:
    1. Clothing item detection
    2. Cropping detected clothing items
    3. Dominant color extraction
    4. Color harmony analysis
    5. Color harmony scoring
    6. Outfit suitability evaluation
    7. Final outfit rating

    Returns:
        A dictionary containing all analysis results.
    """

    # ======================================
    # 1. Clothing Detection
    # ======================================

    # Detect clothing items using the trained YOLO model
    detected_items = detect_clothing(
        image
    )

    # ======================================
    # 2. Crop Detected Items
    # ======================================

    # Crop each detected clothing item
    # from the original image
    cropped_items = crop_detected_items(
        image,
        detected_items
    )

    # ======================================
    # 3. Color Analysis
    # ======================================

    # Analyze the dominant color of each
    # detected clothing item
    analyzed_items = analyze_colors(
        cropped_items
    )

    # ======================================
    # 4. Color Harmony
    # ======================================

    # Determine the overall color harmony
    # of the outfit
    harmony = analyze_color_harmony(
        analyzed_items
    )

    # ======================================
    # 5. Color Score
    # ======================================

    # Convert the detected color harmony
    # into a numerical score from 0 to 10
    color_score = get_color_score(
        harmony
    )

    # ======================================
    # 6. Outfit Suitability
    # ======================================

    # Evaluate how suitable the outfit is
    # for the selected occasion, time, weather,
    # and temperature
    suitability_score, feedback = (
        evaluate_outfit_suitability(
            detected_items,
            occasion,
            time_of_day,
            weather,
            temperature
        )
    )

    # ======================================
    # 7. Final Rating
    # ======================================

    # Calculate the final outfit score.
    #
    # Color harmony contributes 40% of the score,
    # while contextual suitability contributes 60%.
    final_score = (
        color_score * 0.40
        +
        suitability_score * 0.60
    )

    # Round the final score to one decimal place
    final_score = round(
        final_score,
        1
    )

    # ======================================
    # Return Complete Analysis Results
    # ======================================

    return {

        # List of clothing items detected by YOLO
        "detected_items":
            detected_items,

        # Clothing items with their color information
        "analyzed_items":
            analyzed_items,

        # Overall color harmony category
        "harmony":
            harmony,

        # Numerical color harmony score
        "color_score":
            round(color_score, 1),

        # Context-based suitability score
        "suitability_score":
            suitability_score,

        # Final combined outfit rating
        "final_score":
            final_score,

        # Recommendations and warnings
        "feedback":
            feedback
    }