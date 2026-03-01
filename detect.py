# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'./runs/best.pt')
    model.predict(source=r'./datasets\test\images',
                  save=True,
                  #show=True,
                  )
