

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'H:\ultralytics-xx\runs\detect\runs\train\CPU-YOLO\weights\best.pt')
    model.predict(source=r'H:\ultralytics-xx\Datasets\test\images',
                  save=True,
                  #show=True,
                  )
