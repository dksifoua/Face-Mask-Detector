class TrainMaskDetectorConfig:

    DATASET_PATH = './data'
    MODEL_FILEPATH = './checkpoints/mask-detector.model'
    N_EPOCHS = 20
    INIT_LR = 1e-4
    BATCH_SIZE = 32
    PLOT_FILEPATH = './figures/history.png'


class ImageMaskDetectorConfig:

    FACE_DETECTOR_FILEPATH = './checkpoints/mask-detector.model'
    MASK_CLASSIFIER_FILEPATH = './checkpoints/mask-detector.model'
    CONFIDENCE = 0.5
