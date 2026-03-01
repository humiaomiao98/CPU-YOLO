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
                single_cls=False,  # 是否是单类别检测
                batch=8,
                patience=0,
                close_mosaic=30,
                workers=2,
                device='0',
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                 # resume='runs/train/PFAD-YOLO/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                )

