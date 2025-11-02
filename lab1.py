import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sharpen_image_method1(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sharpen_image_method2(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def detect_edges_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    sobel_combined = cv2.magnitude(sobelx, sobely)
    
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    
    sobel_color = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
    
    return sobel_color

def create_custom_filter(image):
    custom_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    
    return cv2.filter2D(image, -1, custom_kernel)

def combine_images(blurred, edges, sharpened, weights):
    total_weight = sum(weights.values())
    
    combined = cv2.addWeighted(
        blurred, weights['blurred']/total_weight,
        edges, weights['edges']/total_weight, 0
    )
    
    combined = cv2.addWeighted(
        combined, 0.7,
        sharpened, weights['sharpened']/total_weight, 0
    )
    
    return combined

def show_images(original, blurred, edges, sharpened, combined, custom_filtered):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Собственный фильтр')
    plt.imshow(cv2.cvtColor(custom_filtered, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_test_image():
    image = np.ones((400, 600, 3), dtype=np.uint8) * 100
    
    cv2.rectangle(image, (50, 50), (200, 150), (255, 0, 0), -1)  
    cv2.circle(image, (400, 100), 60, (0, 255, 0), -1)  
    cv2.putText(image, 'Test Image', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    noise = np.random.randint(0, 50, (400, 600, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def main():
    print("=== Программа обработки изображений ===")
    
    image_path = r'C:\Users\shsgeg\Desktop\face.jpg'
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
    
    print("\nПрименяем фильтры...")
    
    print("1. Медианный фильтр...")
    median_filtered = apply_median_filter(image)
    
    print("2. Гауссово размытие...")
    blurred = apply_gaussian_blur(image)
    
    print("3. Повышение резкости...")
    sharpened_method1 = sharpen_image_method1(image)
    sharpened_method2 = sharpen_image_method2(image)
    
    sharpened = sharpened_method2
    
    print("4. Выделение границ...")
    edges = detect_edges_sobel(image)
    
    print("5. Собственный фильтр...")
    custom_filtered = create_custom_filter(image)
    
    print("6. Комбинирование результатов...")
    weights = {
        'blurred': 0.3,
        'edges': 0.4,
        'sharpened': 0.3
    }
    
    combined = combine_images(blurred, edges, sharpened, weights)
    
    print("\nОтображаем результаты...")
    show_images(image, blurred, edges, sharpened, combined, custom_filtered)
    
    print("Готово!")

if __name__ == "__main__":
    main()