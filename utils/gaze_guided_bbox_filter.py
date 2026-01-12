"""
gaze-guided bounding box filtering
implements hybrid selection: containment-based then distance-based
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


def is_point_in_bbox(point_2d: Tuple[float, float], bbox: List[float]) -> bool:
    """
    check if 2d point is inside bounding box

    args:
        point_2d: (u, v) pixel coordinates
        bbox: [x1, y1, x2, y2]

    returns:
        bool indicating if point is inside bbox
    """
    u, v = point_2d
    x1, y1, x2, y2 = bbox
    return x1 <= u <= x2 and y1 <= v <= y2


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    compute center point of bounding box

    args:
        bbox: [x1, y1, x2, y2]

    returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    compute euclidean distance between two points

    args:
        point1: (x1, y1)
        point2: (x2, y2)

    returns:
        euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def classify_bbox_by_gaze(
    bbox: Dict,
    gaze_2d: Tuple[float, float],
    primary_threshold: float = 100.0
) -> Tuple[str, float]:
    """
    classify bbox as PRIMARY or SECONDARY based on gaze proximity

    primary objects are what the person is actively looking at
    secondary objects provide context

    args:
        bbox: dict with 'bbox' key [x1, y1, x2, y2]
        gaze_2d: (u, v) pixel coordinates where user is looking
        primary_threshold: max pixel distance for PRIMARY classification

    returns:
        tuple of (category, distance_to_gaze)
        category is "PRIMARY" or "SECONDARY"
    """
    center = compute_bbox_center(bbox['bbox'])
    distance = compute_distance(gaze_2d, center)

    # check if gaze is inside bbox (strongest signal)
    if is_point_in_bbox(gaze_2d, bbox['bbox']):
        return ("PRIMARY", 0.0)

    # check if within threshold distance
    if distance <= primary_threshold:
        return ("PRIMARY", distance)

    # everything else is secondary (context)
    return ("SECONDARY", distance)


def select_bbox_by_gaze(
    bboxes: List[Dict],
    gaze_2d: Tuple[float, float],
    max_distance: float = 200.0,
    selection_mode: str = "hybrid"
) -> Optional[Dict]:
    """
    select most relevant bbox based on gaze point

    selection modes:
    - "containment": only select if gaze is inside bbox
    - "distance": select nearest bbox to gaze (within max_distance)
    - "hybrid": try containment first, fallback to distance

    args:
        bboxes: list of dicts with 'score', 'label', 'bbox' keys
        gaze_2d: (u, v) pixel coordinates where user is looking
        max_distance: max pixel distance for distance-based selection
        selection_mode: "containment", "distance", or "hybrid"

    returns:
        selected bbox dict with added 'selection_reason' key, or None
    """
    if not bboxes:
        return None

    u, v = gaze_2d

    # priority 1: check containment
    contained = []
    for bbox in bboxes:
        if is_point_in_bbox(gaze_2d, bbox['bbox']):
            contained.append(bbox)

    if contained and selection_mode in ["containment", "hybrid"]:
        # return highest confidence among contained bboxes
        selected = max(contained, key=lambda x: x['score'])
        selected['selection_reason'] = "gaze_contained"
        selected['distance_to_gaze'] = 0.0
        return selected

    # priority 2: distance-based selection
    if selection_mode in ["distance", "hybrid"]:
        distances = []
        for bbox in bboxes:
            center = compute_bbox_center(bbox['bbox'])
            dist = compute_distance(gaze_2d, center)
            if dist <= max_distance:
                distances.append((dist, bbox))

        if distances:
            # return nearest bbox
            dist, selected = min(distances, key=lambda x: x[0])
            selected['selection_reason'] = "nearest_distance"
            selected['distance_to_gaze'] = float(dist)
            return selected

    # no bbox selected
    return None


def filter_and_rank_bboxes_by_gaze(
    bboxes: List[Dict],
    gaze_2d: Tuple[float, float],
    max_distance: float = 200.0,
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    filter and rank all bboxes by relevance to gaze

    returns bboxes sorted by relevance:
    1. bboxes containing gaze point (sorted by confidence)
    2. bboxes within max_distance (sorted by distance)

    args:
        bboxes: list of bbox dicts
        gaze_2d: (u, v) gaze pixel coordinates
        max_distance: max distance for inclusion
        top_k: if specified, return only top k bboxes

    returns:
        sorted list of bbox dicts with 'distance_to_gaze' added
    """
    if not bboxes:
        return []

    # separate into contained and non-contained
    contained = []
    not_contained = []

    for bbox in bboxes:
        bbox_copy = bbox.copy()
        center = compute_bbox_center(bbox['bbox'])
        dist = compute_distance(gaze_2d, center)
        bbox_copy['distance_to_gaze'] = float(dist)

        if is_point_in_bbox(gaze_2d, bbox['bbox']):
            bbox_copy['selection_reason'] = "gaze_contained"
            contained.append(bbox_copy)
        elif dist <= max_distance:
            bbox_copy['selection_reason'] = "near_gaze"
            not_contained.append(bbox_copy)

    # sort contained by confidence (highest first)
    contained.sort(key=lambda x: x['score'], reverse=True)

    # sort not-contained by distance (nearest first)
    not_contained.sort(key=lambda x: x['distance_to_gaze'])

    # combine: contained first, then nearest
    result = contained + not_contained

    if top_k is not None:
        result = result[:top_k]

    return result


def visualize_gaze_and_bboxes(
    image,
    bboxes: List[Dict],
    gaze_2d: Tuple[float, float],
    selected_bbox: Optional[Dict] = None,
    show_all: bool = True
):
    """
    draw gaze point and bboxes on image

    args:
        image: PIL Image or numpy array
        bboxes: list of all detected bboxes
        gaze_2d: (u, v) gaze coordinates
        selected_bbox: the gaze-selected bbox (highlighted)
        show_all: if true, show all bboxes in gray

    returns:
        annotated image (numpy array)
    """
    import cv2
    from PIL import Image

    # convert to numpy if pil
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    # ensure rgb
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # draw all bboxes in gray
    if show_all:
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(c) for c in bbox['bbox']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 2)
            label = f"{bbox['label']} {bbox['score']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # draw selected bbox in green
    if selected_bbox is not None:
        x1, y1, x2, y2 = [int(c) for c in selected_bbox['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{selected_bbox['label']} (GAZE)"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show selection reason
        reason = selected_bbox.get('selection_reason', 'unknown')
        dist = selected_bbox.get('distance_to_gaze', 0)
        info = f"{reason} (dist: {dist:.1f}px)"
        cv2.putText(img, info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # draw gaze point as red crosshair
    u, v = int(gaze_2d[0]), int(gaze_2d[1])
    crosshair_size = 15
    cv2.line(img, (u - crosshair_size, v), (u + crosshair_size, v), (255, 0, 0), 2)
    cv2.line(img, (u, v - crosshair_size), (u, v + crosshair_size), (255, 0, 0), 2)
    cv2.circle(img, (u, v), 5, (255, 0, 0), -1)
    cv2.putText(img, "GAZE", (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img


if __name__ == "__main__":
    # example usage
    print("gaze-guided bbox filter module")
    print("=" * 50)

    # example bboxes
    bboxes = [
        {'score': 0.85, 'label': 'bottle', 'bbox': [100, 50, 200, 150]},
        {'score': 0.72, 'label': 'cup', 'bbox': [300, 100, 350, 180]},
        {'score': 0.91, 'label': 'hand', 'bbox': [50, 200, 150, 300]},
    ]

    # example gaze point
    gaze_point = (125, 100)  # inside bottle bbox

    print(f"\ngaze point: {gaze_point}")
    print(f"detected {len(bboxes)} objects")

    # test primary/secondary classification
    print(f"\nclassification (primary = actively looking at):")
    for bbox in bboxes:
        category, distance = classify_bbox_by_gaze(bbox, gaze_point, primary_threshold=100.0)
        print(f"  {bbox['label']}: {category} (distance: {distance:.1f}px)")

    # test hybrid mode
    selected = select_bbox_by_gaze(bboxes, gaze_point, selection_mode="hybrid")
    if selected:
        print(f"\nhybrid mode selected: {selected['label']}")
        print(f"  reason: {selected['selection_reason']}")
        print(f"  distance: {selected['distance_to_gaze']:.1f}px")

    # test ranking
    ranked = filter_and_rank_bboxes_by_gaze(bboxes, gaze_point, top_k=2)
    print(f"\ntop 2 ranked by gaze relevance:")
    for i, bbox in enumerate(ranked):
        print(f"  {i+1}. {bbox['label']} ({bbox['selection_reason']}, dist: {bbox['distance_to_gaze']:.1f}px)")
