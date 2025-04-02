import sys
import ast
import os
from PIL import Image

import specified_helper_functions as helper
import inference
from config_class import Config
import hair_color_calculation

# Configure logging
import logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _subtract(coordinate):
    """
    Mainly to check if the coordinates are within the bounds of 
    """
    if coordinate - 20 >= 0:
        return coordinate - 20
    return 0

def _add(coordinate, width_or_height):
    """
    Mainly to check if the coordinates are within the bounds of
    """
    if coordinate + 20 <= width_or_height:
        return coordinate + 20
    return width_or_height

def _enlarge_face_crop(original_img_path, face_crop_bbox):
    temporary_path = ""
    if type(face_crop_bbox) is str:
        face_crop_bbox = ast.literal_eval(face_crop_bbox)
    original_img = Image.open(original_img_path)
    width, height = original_img.size
    new_face_crop_bbox = [
        _subtract(face_crop_bbox[0]),
        _subtract(face_crop_bbox[1]),
        _add(face_crop_bbox[2], width),
        _add(face_crop_bbox[3], height)
    ]
    temporary_path = "data/temporary_enlarged_face.png"
    enlarged_body_crop = original_img.crop(new_face_crop_bbox)
    enlarged_body_crop.save(temporary_path)
    return temporary_path


def _get_face_segmentation():
    args = Config()
    output_path = inference.inference(args)
    return f"{output_path}/temporary_enlarged_face.png"

def _get_hair_color(face_crop_path, segmenteted_face_crop_path):
    hair_color = hair_color_calculation.calculate_hair_color(face_crop_path, segmenteted_face_crop_path)
    return hair_color

def _calculate_pixel_amount(temporary_enlarged_face):
    img = Image.open(temporary_enlarged_face)
    weight, height = img.size
    return weight * height

def hair_color_classification(anon_type):
    # Set up 
    sist_basis = helper.get_anon_type_sist_basis(anon_type)
    all_unedited_cities = helper.collect_all_footage_dfs(anon_type)
    hair_color_data = {}
    for city in all_unedited_cities:
        for index, row in all_unedited_cities[city].iterrows():
            temporary_enlarged_face_path = _enlarge_face_crop(row["original_img_path"], row["face_crop_bbox"])
            temporary_face_segmentation_path = _get_face_segmentation()

            hair_color = _get_hair_color(temporary_enlarged_face_path, temporary_face_segmentation_path)

            hair_color_data[row["person_id"]] = hair_color

            os.remove(temporary_enlarged_face_path)
            os.remove(temporary_face_segmentation_path)


    new_column_data = {
        "hair_color_test": hair_color_data
    }
    
    helper.update_sist_df(sist_basis, new_column_data, anon_type)


# For the case this script is ran as a subprocess
if __name__ == '__main__':
    if len(sys.argv) > 1:
        hair_color_classification(sys.argv[1])
    else:
        logging.warning(f"Age and Gender Classifier weren't able to be executed due to missing anon_type argument!")