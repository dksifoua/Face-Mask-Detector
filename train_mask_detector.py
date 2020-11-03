import os
import tqdm
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import TrainMaskDetectorConfig
from mobile_netv2 import get_model
from logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', action='store', type=str,
        default=TrainMaskDetectorConfig.DATASET_PATH,
        help=f'The path of the dataset. Default: {TrainMaskDetectorConfig.DATASET_PATH}')
    parser.add_argument('--model_filepath', action='store', type=str,
        default=TrainMaskDetectorConfig.MODEL_FILEPATH,
        help=f'The path of the serialized face mask classification model. Default: {TrainMaskDetectorConfig.MODEL_FILEPATH}')
    parser.add_argument('--n_epochs', action='store', type=int,
        default=TrainMaskDetectorConfig.N_EPOCHS,
        help=f'The number of epochs. Default: {TrainMaskDetectorConfig.N_EPOCHS}')
    parser.add_argument('--init_lr', action='store', type=float,
        default=TrainMaskDetectorConfig.INIT_LR,
        help=f'The initial learning rate. Default: {TrainMaskDetectorConfig.INIT_LR}')
    parser.add_argument('--batch_size', action='store', type=int,
        default=TrainMaskDetectorConfig.BATCH_SIZE,
        help=f'The batch size. Default: {TrainMaskDetectorConfig.BATCH_SIZE}')
    parser.add_argument('--plot_filepath', action='store', type=str,
        default=TrainMaskDetectorConfig.PLOT_FILEPATH,
        help=f'The path of the output training history plot. Default: {TrainMaskDetectorConfig.PLOT_FILEPATH}')
    args = parser.parse_args()

    logger = Logger(name='train-mask-detector')

    logger.info('Loading images...')
    data, labels = [], []
    for label in os.listdir(args.dataset_path):
        image_path = os.path.join(args.dataset_path, label)
        for image_filename in tqdm.tqdm(os.listdir(image_path)):
            image = tf.keras.preprocessing.image.load_img(os.path.join(image_path, image_filename))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            data.append(image)
            labels.append(label)
    data = np.array(data, dtype=np.float)
    labels = np.array(labels, dtype=object)
    logger.info(f'Data shape: {data.shape}')
    logger.info(f'Labels shape: {labels.shape}')
    
    logger.info('Binarize labels...')
    binarizer = LabelBinarizer()
    labels = binarizer.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)

    logger.info('Train test split...')
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'X_test shape: {X_test.shape}')
    logger.info(f'Y_train shape: {Y_train.shape}')
    logger.info(f'Y_test shape: {Y_test.shape}')

    logger.infor('Initialize and compile the model')
    model = get_model()
    model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.optimizers.Adam(lr=args.init_lr, decay=args.init_lr / args.n_epochs),
        matrics=[tf.metrics.Accuracy(name='accuracy')])

    logger.info('Build data generation...')
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest')
    
    logger.info('Train the model...')
    history = model.fit(generator.flow(X_train, Y_train, batch_size=args.batch_size),
        steps_per_epoch=len(X_train) // args.batch_size,
        validation_data=(X_test, Y_test),
        validation_steps=len(X_test) // args.batch_size,
        epochs=args.n_epochs)
    
    logger.info('Evaluate the model:')
    probabilities = model.predict(X_test, batch_size=args.batch_size)
    predictions = np.argmax(probabilities, axis=1)
    groung_truths = np.argmax(Y_test, axis=1)
    logger.info(f'{classification_report(groung_truths, predictions, target_names=binarizer.classes_)}')

    logger.info(f'Saving the model to {args.model_filepath}')
    model.save(args.model_filepath, save_format='h5')

    logger.info('Plot model train history')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(args.plot_filepath)
