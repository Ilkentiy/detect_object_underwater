import cv2
import numpy as np

def apply_color_filter(image, filter_color, intensity=0.7):
    """Применяет цветовой фильтр к изображению"""
    color_mask = np.full_like(image, filter_color, dtype=np.uint8)
    filtered_image = cv2.addWeighted(image, 1 - intensity, color_mask, intensity, 0)
    return filtered_image

def parse_filter(color_name):
    """Преобразует название фильтра в BGR цвет"""
    color_map = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'magenta': (255, 0, 255),
        'cyan': (255, 255, 0),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128),
        'none': (0, 0, 0)
    }
    return color_map.get(color_name.lower(), (0, 0, 0))


def color_grader(img, mode='bgr'):
    """Расширенные цветовые преобразования"""
    if mode == 'hsv':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode == 'lab':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif mode == 'gray':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def edge_detector(img, min_thresh=100, max_thresh=200):
    """Детектирование контуров с настройкой параметров"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.Canny(blurred, min_thresh, max_thresh)

def adaptive_segmentation(img, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    """Адаптивная пороговая сегментация"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gray, 255, method, 
                                cv2.THRESH_BINARY, 11, 2)


def kmeans_segmentation(img, K=4):
    """Сегментация методом k-средних"""
    Z = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)