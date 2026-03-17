"""Color utilities: default class colors, display helpers."""

from class_maps.config import DEFAULT_CLASSES, UNCLASSIFIED_COLOR


def get_class_colors(class_definitions):
    """Extract {class_id: (R, G, B)} from class definitions.

    Parameters
    ----------
    class_definitions : dict
        {class_id: {"name": str, "color": (R, G, B)}}

    Returns
    -------
    dict
        {class_id: (R, G, B)}
    """
    return {cid: cdef["color"] for cid, cdef in class_definitions.items()}


def get_next_class_id(class_definitions):
    """Get the next available class ID.

    Parameters
    ----------
    class_definitions : dict
        Current class definitions.

    Returns
    -------
    int
        Next available ID.
    """
    if not class_definitions:
        return 1
    return max(class_definitions.keys()) + 1


def generate_distinct_color(existing_colors):
    """Generate a color that is visually distinct from existing ones.

    Uses a simple hue rotation approach.

    Parameters
    ----------
    existing_colors : list of tuple
        List of (R, G, B) colors already in use.

    Returns
    -------
    tuple
        (R, G, B) new color.
    """
    import colorsys

    n = len(existing_colors)
    # Golden ratio hue stepping for maximum spacing
    hue = (n * 0.618033988749895) % 1.0
    saturation = 0.7
    value = 0.85

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))
