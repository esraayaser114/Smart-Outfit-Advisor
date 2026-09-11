# ==========================================
# IMPORTANT: Model Class Names
# ==========================================

# The model's actual classes are:
# bag, dress, hat, jacket, pants, shirt,
# shoe, shorts, skirt, sunglass
#
# All class names used in the rules below
# must match these names exactly after normalization.
# Otherwise, the corresponding rule will not be triggered.


def normalize_text(value):
    """
    Convert a value to a clean lowercase string.

    This helps make comparisons case-insensitive
    and removes unnecessary spaces.
    """

    return str(value).strip().lower()


def normalize_items(detected_items):
    """
    Extract and normalize clothing item names
    from the detection results.
    """

    # Create a list to store normalized item names
    items = []

    # Process every detected item
    for item in detected_items:

        # If the detection result is a dictionary,
        # get the clothing name from the "item" field
        if isinstance(item, dict):
            name = item.get(
                "item",
                ""
            )

        # Otherwise, convert the item directly to a string
        else:
            name = str(item)

        # Normalize the clothing name and add it to the list
        items.append(
            normalize_text(name)
        )

    return items


def evaluate_outfit_suitability(
    detected_items,
    occasion,
    time_of_day,
    weather,
    temperature
):
    """
    Evaluate how suitable an outfit is based on:

    - Detected clothing items
    - Occasion
    - Time of day
    - Weather
    - Temperature

    Returns:
        A suitability score from 1 to 10
        and a list of feedback messages.
    """

    # Normalize the detected clothing item names
    items = normalize_items(
        detected_items
    )

    # Normalize the contextual inputs
    occasion = normalize_text(
        occasion
    )

    time_of_day = normalize_text(
        time_of_day
    )

    weather = normalize_text(
        weather
    )

    # Convert temperature to a numerical value
    temperature = float(
        temperature
    )

    # Start with the maximum suitability score
    score = 10.0

    # Store recommendations and warnings
    feedback = []

    # ======================================
    # Occasion
    # ======================================

    # Check suitability for work or university
    if occasion in [
        "work",
        "university"
    ]:

        # Shorts may be too casual for these occasions
        if "shorts" in items:

            score -= 3.5

            feedback.append(
                "Shorts may be less suitable "
                "for this occasion."
            )

    # ======================================
    # Sport
    # ======================================

    # Check suitability for sports activities
    if occasion == "sport":

        # Clothing considered too formal for sports
        formal_items = [
            "dress",
            "skirt"
        ]

        # Check whether the outfit contains
        # any formal clothing items
        if any(
            item in items
            for item in formal_items
        ):

            score -= 3.0

            feedback.append(
                "Some items are too formal "
                "for a sporty occasion."
            )

    # ======================================
    # Party
    # ======================================

    # Check suitability for a party
    if occasion == "party":

        # Clothing that may be too casual for a party
        casual_items = [
            "shorts"
        ]

        # Check whether the outfit contains
        # any overly casual items
        if any(
            item in items
            for item in casual_items
        ):

            score -= 2.0

            feedback.append(
                "This item may be too casual "
                "for a party."
            )

    # ======================================
    # Weather
    # ======================================

    # Lightweight clothing items
    light_items = [
        "shorts",
        "skirt"
    ]

    # Heavy clothing items
    # The current model only detects "jacket"
    heavy_items = [
        "jacket"
    ]

    # Outerwear that can provide additional warmth
    outerwear = [
        "jacket"
    ]

    # Determine whether the weather is considered cold
    # based on temperature or the provided weather condition
    cold_condition = (
        temperature < 18
        or weather == "cold"
    )

    # ======================================
    # Cold Weather Rules
    # ======================================

    if cold_condition:

        # Check whether the outfit contains
        # lightweight clothing in cold conditions
        if any(
            item in items
            for item in light_items
        ):

            score -= 2.0

            feedback.append(
                "Some clothing items may "
                "not provide enough warmth."
            )

        # Recommend outerwear if no jacket is detected
        if not any(
            item in items
            for item in outerwear
        ):

            score -= 1.5

            feedback.append(
                "Consider adding a jacket "
                "for extra warmth."
            )

    # ======================================
    # Hot Weather Rules
    # ======================================

    # Determine whether the weather is considered hot
    hot_condition = (
        temperature > 28
        or weather == "hot"
    )

    if hot_condition:

        # A jacket may be uncomfortable in hot weather
        if any(
            item in items
            for item in heavy_items
        ):

            score -= 2.5

            feedback.append(
                "Heavy clothing like a jacket may "
                "be too warm for this temperature."
            )

    # ======================================
    # Rain
    # ======================================

    # Check clothing suitability during rainy weather
    if weather == "rainy":

        # Sunglasses may not be useful during rain
        if "sunglass" in items:

            score -= 0.5

            feedback.append(
                "Sunglasses may be unnecessary "
                "during rainy weather."
            )

    # ======================================
    # Time of Day
    # ======================================

    # Check whether the outfit is being evaluated
    # for evening or nighttime
    if time_of_day in [
        "evening",
        "night"
    ]:

        # Sunglasses are generally less appropriate
        # when there is little or no sunlight
        if "sunglass" in items:

            score -= 1.5

            feedback.append(
                "Sunglasses are usually less "
                "appropriate at night."
            )

    # ======================================
    # Final Score
    # ======================================

    # Keep the final score within the range 1–10
    score = max(
        1.0,
        min(
            10.0,
            score
        )
    )

    # Remove duplicate feedback messages
    feedback = list(
        dict.fromkeys(feedback)
    )

    # If no problems were detected,
    # provide a positive default message
    if not feedback:

        feedback.append(
            "The outfit is well suited "
            "to the selected context."
        )

    # Return the final rounded score and feedback
    return round(score, 1), feedback