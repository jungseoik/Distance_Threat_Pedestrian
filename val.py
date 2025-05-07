from ultralytics import YOLO

model = YOLO("Gender_od/yolov8n-oiv7_openimages_finetune4/weights/best.pt")
name = 'yolo8n-oiv7_test_openimages'
project_dir ="/home/piawsa6000/work/seoik/Classfication_gender_pedestrian/Distance_Threat_Pedestrian/Gender_od"

model.val(
    data="config/gender_od_train_OpenImageData_setting.yaml", 
    split="test",
    project=project_dir,
    name=name,
    exist_ok=True
)
