import cv2
import csv
import datetime
import time
import os
import psutil
import argparse
from ultralytics import YOLO

def process_stream_with_full_stats(model_path, input_video_path, output_csv_path, camera_id, new_width, new_height):
    """
    Эмулирует обработку видеопотока с пропуском кадров и собирает полную статистику
    включая параметры видео, время, память и количество пропущенных кадров.
    """
    # --- Блок инициализации статистики ---
    start_total_time = time.monotonic()
    process = psutil.Process(os.getpid())
    total_frames_processed = 0
    total_frames_skipped = 0
    
    # 1. Загрузка модели YOLO
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # 2. Открытие видеофайла
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {input_video_path}")

    # 3. Получение параметров видео
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if orig_fps == 0:
        orig_fps = 25
    
    print(f"Видеофайл: {os.path.basename(input_video_path)}")
    print(f"Исходные параметры: {orig_width}x{orig_height} @ {orig_fps:.2f} FPS, Всего кадров: {total_frames_in_video}")
    print(f"Параметры обработки: {new_width}x{new_height}")

    # 4. Подготовка CSV-файла
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['timestamp', 'camera_id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'confidence'])
            
            print("\nНачало обработки потока...")
            
            # 5. Основной цикл обработки
            while cap.isOpened():
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                ret, frame = cap.read()
                if not ret or current_frame_pos >= total_frames_in_video:
                    break

                start_time = time.time()
                
                frame_resized = cv2.resize(frame, (new_width, new_height))
                results = model(frame_resized, verbose=False)
                detection_time = datetime.datetime.now().isoformat()
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = round(float(box.conf[0]), 2)
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        csv_writer.writerow([detection_time, camera_id, class_name, x1, y1, x2, y2, confidence])
                
                total_frames_processed += 1
                processing_time = time.time() - start_time
                frames_elapsed_during_processing = processing_time * orig_fps
                frames_to_skip = int(frames_elapsed_during_processing) - 1

                if frames_to_skip > 0:
                    next_frame_to_read = current_frame_pos + frames_to_skip
                    total_frames_skipped += frames_to_skip
                    
                    if next_frame_to_read < total_frames_in_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_to_read)
                    else:
                        break
                
                if total_frames_processed % 50 == 0:
                    print(f"  Обработано: {total_frames_processed} кадров, Пропущено: {total_frames_skipped} кадров...")

    except Exception as e:
        print(f"Произошла ошибка во время обработки: {e}")
    finally:
        # --- Блок сбора и вывода итоговой статистики ---
        end_total_time = time.monotonic()
        total_execution_time = end_total_time - start_total_time
        memory_info = process.memory_info()
        memory_used_mb = memory_info.rss / (1024 ** 2)
        
        cap.release()
        
        print("\n" + "="*45)
        print("              ИТОГОВАЯ СТАТИСТИКА")
        print("="*45)
        
        print("Параметры видео:")
        print(f"  Исходное разрешение:   {orig_width}x{orig_height}")
        print(f"  Разрешение обработки:  {new_width}x{new_height}")
        print(f"  Исходный FPS:            {orig_fps:.2f}")
        print("-" * 45)

        print("Производительность:")
        print(f"  Общее время обработки:   {total_execution_time:.2f} секунд")
        print(f"  Пиковое использование RAM:  {memory_used_mb:.2f} МБ")
        print("-" * 45)

        print("Статистика по кадрам:")
        print(f"  Всего кадров в видео:      {total_frames_in_video}")
        print(f"  Обработано кадров:         {total_frames_processed}")
        print(f"  Пропущено кадров:          {total_frames_skipped}")
        if total_frames_processed > 0:
            avg_time_per_frame = (total_execution_time / total_frames_processed) * 1000
            print(f"  Среднее время на кадр:     {avg_time_per_frame:.2f} мс")
        
        print("="*45)
        print(f"\nОбработка завершена. Результаты сохранены в: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обрабатывает видеофайл с помощью модели YOLO, эмулируя работу в реальном времени с пропуском кадров и сохраняя результаты в CSV."
    )
    
    # Обязательные (позиционные) аргументы
    parser.add_argument(
        "model_path", 
        help="Путь к файлу модели YOLO (например, 'runs/detect/train/weights/best.pt')."
    )
    parser.add_argument(
        "input_video", 
        help="Путь к исходному видеофайлу для обработки."
    )
    parser.add_argument(
        "output_csv", 
        help="Путь для сохранения итогового CSV-файла с детекциями."
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--camera_id", 
        default="CAM-01", 
        help="Идентификатор камеры для записи в CSV. По умолчанию: 'CAM-01'."
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=960, 
        help="Ширина кадра для обработки. По умолчанию: 960."
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=540, 
        help="Высота кадра для обработки. По умолчанию: 540."
    )
    
    args = parser.parse_args()
    
    # Вызов основной функции с параметрами из командной строки
    process_stream_with_full_stats(
        model_path=args.model_path,
        input_video_path=args.input_video,
        output_csv_path=args.output_csv,
        camera_id=args.camera_id,
        new_width=args.width,
        new_height=args.height
    )