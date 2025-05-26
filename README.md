
# Детекция объектов под водой

Разработка интеллектуальной системы компьютерного зрения для автономного необитаемого подводного аппарата

## 📋 Содержание
- [Быстрый старт](#-быстрый-старт)
- [Использование](#-использование)
  - [Изображения](#-детекция-на-изображениях)
  - [Видео](#-детекция-на-видео)
  - [Реальный поток](#-детекция-в-реальном-времени)

## 🚦 Быстрый старт

Установите зависимости:
```pip install -r requirements.txt```
# 🛠 Использование
## 📷 Детекция на изображениях

```python3 scripts/detect_image.py --model runs/detect/train2/weights/best.pt --input predtren/test/test_image_1.png --output predtren/test/detected.png --conf 0.5 ```
  
## 🎥 Детекция на видео

```python3 scripts/detect_video.py --model runs/detect/train2/weights/best.pt --input predtren/test/2019-02-20_19-23-53to2019-02-20_19-24-12_1.avi --output predtren/test/test_video_with_yolo_detection.avi```
  
## ⚡ Детекция в реальном времени

```python3 scripts/detect_stream.py runs/detect/train2/weights/best.pt predtren/test/2019-02-20_19-23-53to2019-02-20_19-24-12_1.avi predtren/test/detection_log.csv```
