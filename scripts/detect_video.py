import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path

def process_video_with_yolo(model_path, input_video_path, output_video_path, new_width=960, new_height=540):
    """
    Обрабатывает видео с детекцией объектов YOLO и сохраняет результат
    
    Args:
        model_path (str): Путь к файлу модели YOLO (.pt)
        input_video_path (str): Путь к входному видеофайлу
        output_video_path (str): Путь для сохранения результата
        new_width (int): Ширина выходного видео (по умолчанию 960)
        new_height (int): Высота выходного видео (по умолчанию 540)
    """
    # Загрузка модели YOLO
    model = YOLO(model_path)
    
    # Открытие видеофайла
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {input_video_path}")
    
    # Получение параметров оригинального видео
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, orig_fps, (new_width, new_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Изменение разрешения кадра
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        # Детекция объектов с помощью YOLO
        results = model(frame_resized)
        
        # Визуализация и запись результатов
        for r in results:
            im_bgr = r.plot()  # Получение кадра с bounding boxes
            out.write(im_bgr)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")
    
    # Освобождение ресурсов
    cap.release()
    out.release()
    print(f"Видео с детекцией сохранено как: {output_video_path}")

if __name__ == "__main__":
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='YOLOv8 Video Object Detection')
    parser.add_argument('--model', type=str, required=True, 
                      help='Путь к файлу модели YOLO (.pt)')
    parser.add_argument('--input', type=str, required=True,
                      help='Путь к входному видеофайлу')
    parser.add_argument('--output', type=str, required=True,
                      help='Путь для сохранения результата')
    parser.add_argument('--width', type=int, default=960,
                      help='Ширина выходного видео (по умолчанию 960)')
    parser.add_argument('--height', type=int, default=540,
                      help='Высота выходного видео (по умолчанию 540)')
    
    args = parser.parse_args()
    
    # Запуск обработки видео
    process_video_with_yolo(
        model_path=args.model,
        input_video_path=args.input,
        output_video_path=args.output,
        new_width=args.width,
        new_height=args.height
    )