import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def get_data_generators():
    """
    Creates and returns data generators for training, validation, and testing.
    Includes Data Augmentation for the training set.
    """
    
    # 1. Augmentation configuration for Training Data
    # This helps the model generalize better by seeing slightly modified versions of images
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalize pixel values to [0,1]
        rotation_range=20,          # Rotate images randomly
        width_shift_range=0.2,      # Shift horizontally
        height_shift_range=0.2,     # Shift vertically
        shear_range=0.2,            # Shear transformations
        zoom_range=0.2,             # Random zoom
        horizontal_flip=True,       # Flip images horizontally
        fill_mode='nearest'
    )

    # 2. Only Rescaling for Validation and Test Data (No augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # 3. Create Generators
    print(f"Loading Training Data from: {TRAIN_DIR}")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',        # 'binary' because we have Bird vs Drone
        shuffle=True
    )

    print(f"Loading Validation Data from: {VAL_DIR}")
    validation_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"Loading Test Data from: {TEST_DIR}")
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator