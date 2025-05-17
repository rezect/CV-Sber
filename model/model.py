from ultralytics import YOLO
import os

def train_model():
    conf_path = "curling_data.yaml"

    # Читаем конфиг
    with open(conf_path, "r") as f:
        data_yaml = f.read()
    
    # Загружаем YOLOv8
    model = YOLO('yolov8n.pt')
    
    # Обучаем модель
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        name='curling_detector'
    )
    
    trained_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    print(f"Обучение завершено. Модель сохранена по пути: {trained_model_path}")
    
    return trained_model_path

def test_on_image(model_path, image_path):
    model = YOLO(model_path)
    
    results = model.predict(image_path, conf=0.25)  # conf - порог уверенности
    
    # Сохраняем результат с визуализацией
    result_path = 'result_detection.jpg'
    results[0].save(result_path)
    
    print(f"Результат сохранен в файл {result_path}")
    return results

if __name__ == "__main__":
    # Шаг 1: Обучение модели
    # model_path = train_model()
    
    # Шаг 2: Тестирование модели на одном изображении
    model_path = 'D:/Programing/Not university/CV Sber/runs/detect/curling_detector/weights/best.pt'
    test_image = 'frames/images/frame_000540.jpg'
    
    results = test_on_image(model_path, test_image)