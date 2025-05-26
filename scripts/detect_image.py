import argparse
from ultralytics import YOLO
import cv2

def process_image(model_path, input_path, output_path, conf=0.5):
    """
    Обрабатывает изображение с помощью YOLO модели и сохраняет результат
    
    Args:
        model_path (str): Путь к файлу модели (.pt)
        input_path (str): Путь к входному изображению
        output_path (str): Путь для сохранения результата
        conf (float): Порог уверенности для детекции (0-1)
    """
    # Загружаем модель
    model = YOLO(model_path)
    
    # Выполняем предсказание
    results = model.predict(source=input_path, conf=conf)
    
    # Получаем первое изображение с результатами (если batch=1)
    plotted = results[0].plot()
    
    # Сохраняем результат
    cv2.imwrite(output_path, plotted)
    print(f"Результат сохранен в {output_path}")

if __name__ == "__main__":
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу модели (.pt)')
    parser.add_argument('--input', type=str, required=True, help='Путь к входному изображению')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения результата')
    parser.add_argument('--conf', type=float, default=0.5, help='Порог уверенности (0-1)')
    
    args = parser.parse_args()
    
    # Запускаем обработку
    process_image(args.model, args.input, args.output, args.conf)