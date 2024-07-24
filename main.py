from ultralytics import YOLOv10
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    model = YOLOv10('LD-YOLOv10.yaml')  #
    model.train(**{'cfg': 'default.yaml', 'data': 'dataset/data.yaml'})



