import numpy as np
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt
import cv2
import os

def segment_matriz(matriz, limit=10):
    m, n = matriz.shape
    regions = -np.ones_like(matriz, dtype=int)
    cur_region = 0

    def neighborhood(x, y):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n:
                yield nx, ny

    for i in range(m):
        for j in range(n):
            if regions[i, j] == -1:
                # Start a new region.
                queue = deque()
                queue.append((i, j))
                regions[i, j] = cur_region

                while queue:
                    x, y = queue.popleft()
                    for nx, ny in neighborhood(x, y):
                        dif = abs(int(matriz[nx, ny]) - int(matriz[x, y]))
                        if regions[nx, ny] == -1 and dif < limit:
                            regions[nx, ny] = cur_region
                            queue.append((nx, ny))

                cur_region += 1

    return regions

def save_segmented_image(imageName, image):
    normalized = (image / image.max() * 255).astype('uint8')

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    cv2.imwrite(imageName, colored)
    return

my_canny_filenames = os.listdir('my_result_with_canny')
original_filenames = os.listdir('input')

for image_name in my_canny_filenames:  
    image_my_result_with_canny = cv2.imread("my_result_with_canny/"+image_name, cv2.IMREAD_GRAYSCALE)

    my_canny_segments = segment_matriz(image_my_result_with_canny)

    save_segmented_image("segmentation/" + "segmentation_" + image_name, my_canny_segments)

for image_name in original_filenames:  
    image_original = cv2.imread("input/"+image_name, cv2.IMREAD_GRAYSCALE)

    original_segments = segment_matriz(image_original)

    save_segmented_image("segmentation/" + "segmentation_" + image_name, original_segments)



## Create a 2x2 grid of subplots
#fig, axs = plt.subplots(2, 2, figsize=(8, 8))

## Plot each image
#axs[0, 0].imshow(image_original, cmap='gray')
#axs[0, 0].set_title('imagem original')
#axs[0, 0].axis('off')

#axs[0, 1].imshow(original_segments, interpolation='none', cmap='gray')
#axs[0, 1].set_title('Segmentacao')
#axs[0, 1].axis('off')

#axs[1, 0].imshow(canny_segments, interpolation='none', cmap='gray')
#axs[1, 0].set_title('Canny e Segmentacao')
#axs[1, 0].axis('off')

#axs[1, 1].imshow(my_canny_segments, interpolation='none', cmap='gray')
#axs[1, 1].set_title('MyAlgorithm, Canny e Segmentacao')
#axs[1, 1].axis('off')

## Adjust layout
#plt.tight_layout()
#plt.show()