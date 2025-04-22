from ultralytics import YOLO

model = YOLO("Gender_cls/ai_hub2/weights/last.pt")
model.train(cfg="config/gender_cls_aihub.yaml")