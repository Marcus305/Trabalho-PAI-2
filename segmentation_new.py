import numpy as np
from PIL import Image
from collections import deque
import cv2
import os

def segment_matriz(matriz, limit=10):
    """
    Realiza a segmentação de uma imagem (matriz numpy) usando um algoritmo de crescimento de região.
    """
    if matriz is None:
        print("Erro: Matriz de entrada está vazia (None).")
        return None

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
                # Inicia uma nova região
                queue = deque()
                queue.append((i, j))
                regions[i, j] = cur_region

                while queue:
                    x, y = queue.popleft()
                    for nx, ny in neighborhood(x, y):
                        # Garante que os valores da matriz sejam tratados como inteiros para o cálculo da diferença
                        dif = abs(int(matriz[nx, ny]) - int(matriz[x, y]))
                        if regions[nx, ny] == -1 and dif < limit:
                            regions[nx, ny] = cur_region
                            queue.append((nx, ny))
                
                cur_region += 1

    return regions

def save_segmented_image(imageName, image):
    """
    Salva a matriz de regiões segmentadas como uma imagem colorida.
    """
    if image is None:
        print(f"Aviso: Nenhuma imagem para salvar em {imageName}.")
        return

    # Normaliza a imagem para o intervalo 0-255 para visualização
    max_val = image.max()
    if max_val == 0:
        normalized = image.astype('uint8') # Evita divisão por zero se a imagem for toda preta
    else:
        normalized = (image / max_val * 255).astype('uint8')

    # Aplica um mapa de cores para que cada segmento tenha uma cor diferente
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Salva a imagem resultante
    cv2.imwrite(imageName, colored)
    return

# --- Lógica Principal Modificada ---

# Define a pasta de entrada com base na sua solicitação.
# O caminho é relativo ao local onde o script `segmentation_fael.py` está.
input_folder = './Sobel/sobel_edge_detection_output'

# Define a pasta de saída para as imagens segmentadas.
# Uma nova pasta será criada se não existir.
output_folder = './Sobel/saidas_sobel_segmentadas'

print(f"Processando pasta de entrada: '{input_folder}'")
print(f"Os resultados serão salvos em: '{output_folder}'")

# 1. Cria o diretório de saída se ele não existir
os.makedirs(output_folder, exist_ok=True)

# 2. Verifica se a pasta de entrada existe
if not os.path.isdir(input_folder):
    print(f"ERRO: A pasta de entrada '{input_folder}' não foi encontrada. Verifique o caminho e a estrutura de pastas.")
else:
    # 3. Lista todos os arquivos na pasta de entrada
    try:
        filenames = os.listdir(input_folder)
        if not filenames:
            print(f"Aviso: Nenhuma imagem encontrada em '{input_folder}'.")
        else:
            # 4. Processa cada arquivo
            for image_name in filenames:
                input_path = os.path.join(input_folder, image_name)
                output_path = os.path.join(output_folder, image_name)

                # Garante que estamos processando apenas arquivos
                if os.path.isfile(input_path):
                    # Lê a imagem em escala de cinza
                    image_matrix = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                    if image_matrix is not None:
                        # Executa o algoritmo de segmentação
                        # Você pode ajustar o 'limit' aqui se precisar
                        segmented_regions = segment_matriz(image_matrix, limit=10)

                        # Salva a imagem segmentada e colorida
                        save_segmented_image(output_path, segmented_regions)
                        print(f"  -> Imagem '{image_name}' processada e salva em '{output_path}'")
                    else:
                        print(f"Aviso: Não foi possível ler o arquivo de imagem: {input_path}")

    except FileNotFoundError:
        print(f"Erro ao acessar a pasta '{input_folder}'.")

print("\nProcessamento concluído!")