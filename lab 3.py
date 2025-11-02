import cv2
import numpy as np
import time

class SocialRobot:
    def __init__(self):
        # Загрузка каскадов для обнаружения
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Состояния пользователя
        self.smiling = False
        self.eyes_open = False
        self.face_detected = False
        self.message = "Подойдите ближе к камере"
        
        # Переменные для FPS
        self.prev_time = 0
        self.fps = 0
        self.frame_count = 0
        
    def calculate_fps(self):
        """Вычисление FPS"""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        
        self.prev_time = current_time
        return self.fps

    def detect_face_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        self.smiling = False
        self.eyes_open = False
        self.face_detected = len(faces) > 0
        self.message = ""
        
        if self.face_detected:
            for (x, y, w, h) in faces:
                # Рисуем прямоугольник вокруг лица
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Область лица для поиска глаз и улыбки
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Обнаружение улыбки
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(25, 25)
                )
                
                if len(smiles) > 0:
                    self.smiling = True
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                        cv2.putText(roi_color, "Smile", (sx, sy-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Обнаружение глаз
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                open_eyes_count = 0
                for (ex, ey, ew, eh) in eyes:
                    # Проверяем, что это действительно глаза (расположены в верхней части лица)
                    if ey < h/2:
                        open_eyes_count += 1
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                        cv2.putText(roi_color, "Eye", (ex, ey-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                self.eyes_open = (open_eyes_count >= 2)
                
                # Определяем сообщение для пользователя
                if not self.smiling and not self.eyes_open:
                    self.message = "Улыбнись и открой глаза"
                elif not self.smiling:
                    self.message = "Улыбнись"
                elif not self.eyes_open:
                    self.message = "Открой глаза"
                else:
                    self.message = "Отлично! Приветствую!"
                    
                # Добавляем информацию о состоянии
                status_text = f"Smile: {'YES' if self.smiling else 'NO'}, Eyes: {'OPEN' if self.eyes_open else 'CLOSED'}"
                cv2.putText(frame, status_text, (x, y-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        else:
            self.message = "Лицо не обнаружено. Подойдите ближе"
            
        return frame

    def display_interface(self, frame):
        """Создание интерфейса с сообщениями робота и FPS"""
        # Создаем копию кадра для интерфейса
        interface_frame = frame.copy()
        
        # Добавляем фон для текста
        overlay = interface_frame.copy()
        cv2.rectangle(overlay, (0, 0), (interface_frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, interface_frame, 0.4, 0, interface_frame)
        
        # Отображаем FPS в левом верхнем углу
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(interface_frame, fps_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Сообщение робота
        cv2.putText(interface_frame, "Робот:", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Цвет сообщения в зависимости от состояния
        message_color = (0, 255, 0) if (self.smiling and self.eyes_open) else (0, 165, 255)
        cv2.putText(interface_frame, self.message, (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, message_color, 2)
        
        # Инструкция для пользователя
        cv2.putText(interface_frame, "Нажмите 'q' для выхода", 
                   (10, interface_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return interface_frame

def main():
    robot = SocialRobot()
    cap = cv2.VideoCapture(0)
    
    # Устанавливаем начальное время для FPS
    robot.prev_time = time.time()
    
    print("Социальный робот активирован!")
    print("Инструкции:")
    print("- Подойдите к камере")
    print("- Улыбнитесь")
    print("- Откройте глаза")
    print("- Нажмите 'q' для выхода")
    print("- FPS отображается в левом верхнем углу")
    
    frame_count = 0
    start_time = time.time()
    
    # Создаем окно с фиксированным названием
    window_name = 'Social Robot Interface'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка получения кадра с камеры")
            break
        
        frame_count += 1
        
        # Вычисляем FPS
        robot.calculate_fps()
        
        # Обрабатываем кадр
        processed_frame = robot.detect_face_features(frame)
        
        # Создаем интерфейс
        interface_frame = robot.display_interface(processed_frame)
        
        # Отображаем результат в том же окне
        cv2.imshow(window_name, interface_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Вычисляем средний FPS за все время работы
    end_time = time.time()
    total_time = end_time - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nСтатистика работы:")
    print(f"Общее время работы: {total_time:.2f} секунд")
    print(f"Обработано кадров: {frame_count}")
    print(f"Средний FPS: {average_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Работа робота завершена")

if __name__ == "__main__":
    main()