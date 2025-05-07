from ultralytics import YOLO

model = YOLO("yolov8n-oiv7.pt")
# model = YOLO("yolo11s.pt")

model.train(cfg="config/gender_od_train_OpenImageData.yaml")
