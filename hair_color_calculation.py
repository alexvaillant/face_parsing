import cv2
import numpy as np

RGB_HAIR_COLOR = {    
    "true_black": (9, 8, 6),    
    "brown": (106, 78, 66),    
    "blond": (230, 206, 168),    
    "gray": (113, 99, 90),    
    "white": (255, 24, 225),    
    "red": (181, 82, 57)
}


def _euclidean_distance(point_1, point_2):
    added = 0    
    i = 0
    while i < len(point_1):
        added += (point_1[i] - point_2[i]) ** 2

    dist = np.sqrt(added)
    return dist


def _calculate_closest(hair_vector):
    if hair_vector == [0, 0, 0]:
        return "black_or_bald"
    
    if len(hair_vector) != 3:
        return None
    
    current_closest = RGB_HAIR_COLOR["true_black"]
    closest_color = "true_black"
    for color in RGB_HAIR_COLOR:
        if color == "true_black":
            continue
        rgb_values = RGB_HAIR_COLOR[color]
        if _euclidean_distance(hair_vector, rgb_values) < _euclidean_distance(hair_vector, current_closest):
            current_closest = rgb_values
            closest_color = color

    return closest_color


def _calculate_hair_color_vector(face_crop_path, segmenteted_face_crop_path):
    segmented_image = cv2.imread(segmenteted_face_crop_path)
    original_image = cv2.imread(face_crop_path)

    # Convert images to RGB format
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    hair_color_segment = np.array([240, 220, 180])

    # Create a mask for hair pixels by finding pixels close to the hair color
    hair_mask = np.all(np.abs(segmented_image - hair_color_segment) < 30, axis=-1)

    # Extract hair pixels from the original image
    hair_pixels = original_image[hair_mask]

    mean_hair_color = np.mean(hair_pixels, axis=0) if len(hair_pixels) > 0 else [0, 0, 0]

    if mean_hair_color is np.array:
        mean_hair_color = mean_hair_color.to_list()

    return mean_hair_color


def calculate_hair_color(face_crop_path, segmenteted_face_crop_path):
    hair_color_vector =_calculate_hair_color_vector(face_crop_path, segmenteted_face_crop_path)
    hair_color = _calculate_closest(hair_color_vector)
    return hair_color