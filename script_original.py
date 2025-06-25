from PIL import Image
import numpy as np
import os

path = "input\image2.jpg"
base_name = os.path.basename(path)
fileName = os.path.splitext(base_name)[0]

filtro4 = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

filtro3 = [
    [0, -1, 0],
    [-1,  4, -1],
    [0, -1, 0]
]

filtro2 = [
    [1, 1, 1],
    [1,  8, 1],
    [1, 1, 1]
]

filtro1 = [
    [0, 1, 0],
    [1,  4, 1],
    [0, 1, 0]
]

def filtro_mediana(imagem, tamanho_janela=5):
    offset = tamanho_janela // 2
    altura, largura = imagem.shape
    resultado = np.copy(imagem)

    for i in range(offset, altura - offset):
        for j in range(offset, largura - offset):
            vizinhos = imagem[i-offset:i+offset+1, j-offset:j+offset+1].flatten()
            resultado[i, j] = np.median(vizinhos)
    
    return resultado.astype(np.uint8)

# --- 1. Carregamento e pré-processamento da imagem ---
img = Image.open(path).convert('L')
matriz_original = np.array(img).astype(np.float32)
altura, largura = matriz_original.shape

# Calcula a média
media = np.mean(matriz_original)

# Indica os ajustes a serem feitos
print(f"Média dos tons de cinza: {media:.2f}")
if media < 50:
    print("Imagem escura: usar A = 10.0 e limiar_diferenca = 90")
    A = 10.0
    limiar_diferenca = 90
elif media < 170:
    print("Imagem média: usar A = 2.0 e limiar_diferenca = 80")
    A = 2.0
    limiar_diferenca = 80
else:
    print("Imagem clara: usar A = 1.4 e limiar_diferenca = 9")
    A = 1.4
    limiar_diferenca = 9


# --- 2. Aplica suavização Gaussiana 3x3 ---
kernel = np.array([[1, 4, 1],
                   [4, 8, 4],
                   [1, 4, 1]], dtype=np.float32)
kernel = kernel / kernel.sum()

suavizada = np.zeros_like(matriz_original)

for i in range(1, altura - 1):
    for j in range(1, largura - 1):
        região = matriz_original[i-1:i+2, j-1:j+2]
        suavizada[i, j] = np.sum(região * kernel)

# --- 3. Filtro High-Boost ---
matriz_original = filtro_mediana(matriz_original)
if media < 50:
    suavizada = filtro_mediana(suavizada, tamanho_janela=10)
else:
    suavizada = filtro_mediana(suavizada, tamanho_janela=9)
highboost = (A * matriz_original) - suavizada
highboost = np.clip(highboost, 0, 255).astype(np.uint8)

# --- 4. Detecção de bordas com Laplaciano ---
laplaciano = np.zeros((altura, largura), dtype=np.int32)
# Máscara Laplaciana 8-conectada
mascara = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Aplicando o filtro
for i in range(1, altura - 1):
    for j in range(1, largura - 1):
        regiao = highboost[i-1:i+2, j-1:j+2]
        valor = np.sum(regiao * mascara)
        laplaciano[i, j] = valor

contorno = np.where(np.abs(laplaciano) > limiar_diferenca, 255, 0).astype(np.uint8)

# --- 6. Salva os resultados ---
Image.fromarray(highboost).save(fileName + '_highboost' + '.png')
Image.fromarray(contorno).save(fileName + '_contorno' + '.png')

# --- 7. Mostra visualmente ---
Image.fromarray(highboost).show(title='High-Boost')
Image.fromarray(contorno).show(title='Contorno Detectado')