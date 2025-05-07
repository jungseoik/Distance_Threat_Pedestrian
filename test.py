from ultralytics import YOLO
from pathlib import Path


project_dir = Path("/home/piawsa6000/work/seoik/Classfication_gender_pedestrian/Distance_Threat_Pedestrian/Gender_od")
project_dir.mkdir(parents=True, exist_ok=True)
name = 'yolo_openimages'
model_path = 'yolov8n-oiv7.pt'
data_yaml = 'config/gender_od_test_OpenImageData_opid.yaml'

print(f'\n🔍 Evaluating: {name}')
model = YOLO(model_path)

metrics = model.val(
    data=data_yaml,
    split='test',
    imgsz=640,
    save=True,
    project=str(project_dir),
    name=name,
    exist_ok=True
)

print(f"📊 {name} Results:")
print(f"  mPrecision:   {metrics.box.mp:.4f}")
print(f"  mRecall:      {metrics.box.mr:.4f}")
print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
