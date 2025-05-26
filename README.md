





Детектирвоание объектов на изображении
python3 scripts/detect_image.py --model
runs/detect/train2/weights/best.pt --inputpredtren/test/test_image_1.png --output predtren/test/detected.png
--conf 0.5


Детектирование объектов на видео
python3
scripts/detect_video.py --model runs/detect/train2/weights/best.pt--input predtren/test/2019-02-20_19-23-53to2019-02-20_19-24-12_1.avi
--output predtren/test/test_video_with_yolo_detection.avi


Детектирование объектов на видео в режиме реального потока
python3
scripts/detect_stream.py runs/detect/train2/weights/best.ptpredtren/test/2019-02-20_19-23-53to2019-02-20_19-24-12_1.avi
predtren/test/detection_log.csv
