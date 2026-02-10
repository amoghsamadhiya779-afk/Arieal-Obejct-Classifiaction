import os
import tensorflow as tf
from src.config import EPOCHS, LEARNING_RATE, MODEL_DIR
from src.data_loader import get_data_generators
from src.model_builder import build_custom_cnn, build_transfer_learning_model

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(model_type='custom'):
    """
    Main training loop.
    :param model_type: 'custom' or 'transfer'
    """
    
    # 1. Get Data
    train_gen, val_gen, _ = get_data_generators()

    # 2. Build Model
    if model_type == 'custom':
        model = build_custom_cnn()
        filename = 'custom_cnn_aerial.h5'
    else:
        model = build_transfer_learning_model()
        filename = 'transfer_mobilenet_aerial.h5'

    print(f"\nTraining {model.name}...")
    model.summary()

    # 3. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, filename),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # 5. Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    print(f"Training Complete. Model saved to {os.path.join(MODEL_DIR, filename)}")
    return history

if __name__ == "__main__":
    # Choose which model to train
    # train_model('custom')
    train_model('transfer')