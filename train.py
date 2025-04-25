from ultralytics import YOLO

model = YOLO("yolov8m-cls.pt")
model.train(cfg="config/gender_cls_aihub.yaml")