
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('./yolov8n.pt')
    # 开始训练
    model.train(
        data='./myData.yaml',  # 指向 data.yaml 文件的路径
        epochs=100,  # 训练的轮次
        batch=32,  # 每次训练的批量大小
        imgsz=640,  # 输入图像尺寸
        device=0,  # 使用的设备：0 表示使用 GPU，'cpu' 表示使用 CPU
        optimizer='Adam'
    )

    # 保存训练好的模型
    model.save('./models/best_model1.pt')

