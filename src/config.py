params = {
    # DISPLAY PROPERTIES =================
    "WindowPosX": 200,
    "WindowPosY": 200,
    # CV PARAMETERS ======================
    # TAG DETECTION ----------------------
    "HoughTolerance": 25,                       # Higher value detects less circles
    "ColorThreshold": 0.35,                     # Threshold for labeling colors
    # TAG PROPERTIES ---------------------
    "TagSizeScreenHeightRatio": 0.1,#0.05,
    "TagSizeMaxPercentage": 1.2,
    "TagSizeMinPercentage": 0.8,
    "MinTagDist": 10,
    "ColorDetectSamples": 25,                   # Higher values for accurate color detection at the cost of performance
    # COLORS =============================
    "YellowMin": [15, 30, 150],
    "YellowMax": [50, 150, 255],
    "WhiteMin": [0, 0, 200],
    "WhiteMax": [180, 25, 255],
    # CHUNKING ===========================
    "CropSize": 200,
    # OCR PARAMETERS =====================
    "DigitContourArea": 400,
    "DigitHeight": 25,
    "RotDigitContourArea": 200,
    "TrainingDigitContourArea": 85,
    "TrainingDigitHeight": 30,
    "FeatureVectorSize": (20, 20),
    "kNearest": 4,
    "TagSizeCropFactor": 0.95,
    "AverageTagRadius": 15
}
