import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'./ultralytics/cfg/models/CPU-YOLO.yaml')
    model.train(data=r"./datasets/data.yaml",
                task='train',
                cache=False,
                imgsz=640,
                epochs=600,
                single_cls=False,
                batch=8,
                patience=0,
                close_mosaic=30,
                workers=2,
                device='0',
                optimizer='SGD',
                 # resume='runs/train/xx-YOLO/weights/last.pt',
                amp=True,
                )

