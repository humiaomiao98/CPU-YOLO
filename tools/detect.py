

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'H:\ultralytics-main\runs\detect\runs\train\CPU-YOLO\weights\best.pt')
    model.predict(source=r'H:\ultralytics-main\Datasets\test\images',
                  save=True,
                  #show=True,
                  )
