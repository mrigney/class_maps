"""Global configuration: default classes, SLIC/RF parameters, constants."""

from collections import OrderedDict


# Default landcover classes: {id: {"name": str, "color": (R, G, B)}}
# ID 0 is reserved for "unclassified"
DEFAULT_CLASSES = OrderedDict({
    1:  {"name": "Coniferous forest",  "color": (0, 100, 0)},
    2:  {"name": "Deciduous forest",   "color": (34, 139, 34)},
    3:  {"name": "Shrub/scrub",        "color": (128, 128, 0)},
    4:  {"name": "Grass/field",        "color": (144, 238, 144)},
    5:  {"name": "Agriculture",        "color": (173, 255, 47)},
    6:  {"name": "Bare soil",          "color": (210, 180, 140)},
    7:  {"name": "Road (paved)",       "color": (105, 105, 105)},
    8:  {"name": "Road (unpaved)",     "color": (185, 156, 107)},
    9:  {"name": "Water",              "color": (65, 105, 225)},
    10: {"name": "Structure/building", "color": (220, 20, 60)},
})

UNCLASSIFIED_COLOR = (50, 50, 50)

# SLIC superpixel defaults
SLIC_N_SEGMENTS = 2000
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 1.0

# GLCM texture parameters
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, 0.785, 1.571, 2.356]  # 0, 45, 90, 135 degrees

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 2

# Class IDs eligible for canopy density computation
VEGETATION_CLASS_IDS = [1, 2, 3]

# Supported file extensions
GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
PNG_EXTENSIONS = {".png"}
SUPPORTED_EXTENSIONS = GEOTIFF_EXTENSIONS | PNG_EXTENSIONS

# Shadow detection thresholds (HSV V channel, 0-255 scale)
SHADOW_V_THRESHOLD = 60
SHADOW_S_THRESHOLD = 40
