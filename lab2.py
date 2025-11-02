import cv2
import numpy as np

def simple_face_sketch(image_path):
    """Простая версия преобразования лица в рисунок"""
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return
    
    # Конвертация в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Улучшение контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Применение фильтра Канни
    edges = cv2.Canny(blurred, 50, 150)
    
    # Инверсия для создания рисунка (черные линии на белом фоне)
    sketch = cv2.bitwise_not(edges)
    
    # Изменяем размеры изображений для объединения
    height, width = image.shape[:2]
    
    # Создаем белый фон для объединенного изображения
    combined = np.ones((height, width * 3, 3), dtype=np.uint8) * 255
    
    # Размещаем оригинальное изображение
    combined[0:height, 0:width] = image
    
    # Размещаем границы Канни (конвертируем в BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined[0:height, width:width*2] = edges_bgr
    
    # Размещаем графический рисунок (конвертируем в BGR)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    combined[0:height, width*2:width*3] = sketch_bgr
    
    # Добавляем подписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (50, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(combined, 'Canny Edges', (width + 50, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(combined, 'Face Sketch', (width*2 + 50, 30), font, 1, (0, 0, 0), 2)
    
    # Отображаем объединенное изображение
    cv2.imshow('Face Sketch Results - Press any key to close', combined)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Использование
image_path = r'C:\Users\shsgeg\Desktop\face.jpg'
simple_face_sketch(image_path)