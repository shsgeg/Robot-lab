import cv2
import numpy as np

# Настройки для синего цвета в HSV
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([130, 255, 200])

# Минимальная площадь
MIN_AREA = 500

def detect_blue_objects():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к камере")
        return
    
    print("Запуск обнаружения синих объектов...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр с камеры")
            break
        
        # Конвертируем BGR в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создание маски для синего цвета
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        
        # Морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        blue_objects_count = 0
        
        # Обработка контуров
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > MIN_AREA:
                blue_objects_count += 1
                
                # Определение ограничивающего прямоугольника
                rect = cv2.minAreaRect(contour)
                center = rect[0]
                size = rect[1]
                angle = rect[2]
                
                # Получаем координаты углов прямоугольника
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Расчет углов
                angle_horizontal = -angle if angle < 0 else angle
                angle_vertical = 90 - angle_horizontal
                
                # Отрисовка результатов
                
                # Рисуем контур (зеленый)
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Рисуем минимальный прямоугольник (красный)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                
                # Рисуем центр (синяя точка)
                center_x, center_y = int(center[0]), int(center[1])
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # Добавляем информацию об объекте
                cv2.putText(frame, f"Blue {blue_objects_count}", 
                           (center_x - 20, center_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Вывод информации в консоль
                print(f"Синий объект {blue_objects_count}: Центр ({center_x}, {center_y}), " +
                      f"Угол: {angle_horizontal:.1f}°, Площадь: {area:.0f} px")
        
        # Общая информация на кадре
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Blue objects detected: {blue_objects_count}", 
                   (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, frame.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
        
        # Отображение результатов
        cv2.imshow('Blue Object Detection', frame)
        cv2.imshow('Blue Mask', mask)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")

# Запуск программы
if __name__ == "__main__":
    detect_blue_objects()