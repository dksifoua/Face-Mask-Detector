import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

from config import ImageMaskDetectorConfig
from logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filepath', action='store', type=str, required=True,
        help=f'The path of the image file.')
    parser.add_argument('--face_detector_model_filepath', action='store', type=str,
        default=ImageMaskDetectorConfig.FACE_DETECTOR_FILEPATH,
        help=f'The path of the serialized face detector classification model. Default: {ImageMaskDetectorConfig.FACE_DETECTOR_FILEPATH}')
    parser.add_argument('--mask_detector_model_filepath', action='store', type=str,
        default=ImageMaskDetectorConfig.MASK_CLASSIFIER_FILEPATH,
        help=f'The path of the serialized face mask classification model. Default: {ImageMaskDetectorConfig.MASK_CLASSIFIER_FILEPATH}')
    parser.add_argument('--confidence', action='store', type=float,
        default=ImageMaskDetectorConfig.CONFIDENCE,
        help=f'The minimum probability to filter weak detections. Default: {ImageMaskDetectorConfig.CONFIDENCE}')
    args = parser.parse_args()

    logger = Logger(name='detect-mask-image')

    logger.info('Loading face detector model...')
    
