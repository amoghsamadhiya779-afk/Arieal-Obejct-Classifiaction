import tensorflow as tf
from tensorflow.keras import layers, models, applications
from src.config import IMG_SIZE, CHANNELS

def build_custom_cnn():
    """
    Constructs a custom Convolutional Neural Network.
    """
    model = models.Sequential([
        # Input Layer
        layers.InputLayer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer (Sigmoid for Binary Classification)
        layers.Dense(1, activation='sigmoid')
    ], name="Custom_CNN_Aerial")
    
    return model

def build_transfer_learning_model(base_arch='MobileNetV2'):
    """
    Constructs a model using Transfer Learning (e.g., MobileNetV2).
    """
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    
    if base_arch == 'MobileNetV2':
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet'
        )
    else:
        # Fallback to MobileNetV2 or add other options like ResNet50
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet'
        )

    # Freeze the base model to keep pre-trained weights
    base_model.trainable = False

    # Add custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name=f"Transfer_{base_arch}")

    return model