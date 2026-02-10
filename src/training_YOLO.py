from ultralytics import YOLO
from src.config import YOLO_YAML_PATH, MODEL_DIR, EPOCHS

def train_yolo_model():
    """
    Trains a YOLOv8 Nano model on the aerial dataset.
    Requires the 'ultralytics' library.
    """
    print("Initializing YOLOv8 Nano model...")
    # Load a pretrained model (recommended for faster convergence)
    model = YOLO("yolov8n.pt") 

    print(f"Starting Training using config: {YOLO_YAML_PATH}")
    
    # Train the model
    # project=MODEL_DIR ensures the results are saved in our main models folder
    results = model.train(
        data=YOLO_YAML_PATH,
        epochs=EPOCHS,
        imgsz=640,
        project=MODEL_DIR,
        name='yolov8_aerial_detection',
        patience=10
    )
    
    print("YOLO Training Completed.")
    print(f"Best model saved at: {MODEL_DIR}/yolov8_aerial_detection/weights/best.pt")

if __name__ == "__main__":
    train_yolo_model()