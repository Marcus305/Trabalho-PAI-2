import cv2
from PIL import Image
import os

pathI = "input\image2.jpg"
pathO = '.\image2_highboost.png'
base_name = os.path.basename(pathI)
fileName = os.path.splitext(base_name)[0]

def contar_e_desenhar_objetos(imagem_original_path, imagem_bordas_path, output_path):
    # Carrega a imagem original para desenhar sobre ela
    img_original = cv2.imread(imagem_original_path)
    # Carrega a imagem de bordas (já em escala de cinza)
    img_bordas = cv2.imread(imagem_bordas_path, cv2.IMREAD_GRAYSCALE)

    # Encontra os contornos
    # cv2.RETR_EXTERNAL é bom para pegar apenas os contornos externos dos objetos
    # cv2.CHAIN_APPROX_SIMPLE economiza memória
    contornos, _ = cv2.findContours(img_bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos muito pequenos (que provavelmente são ruído)
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > 50] # ajuste o valor 50

    # Desenha os contornos na imagem original
    cv2.drawContours(img_original, contornos_filtrados, -1, (0, 255, 0), 2) # Desenha em verde

    # Escreve o número de objetos encontrados
    texto = f"Objetos encontrados: {len(contornos_filtrados)}"
    cv2.putText(img_original, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Salva o resultado
    cv2.imwrite(output_path, img_original)
    Image.fromarray(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)).show()

# Exemplo de uso para cada método
contar_e_desenhar_objetos(pathI, 'resultado_contorno.png', 'contagem_metodo_original.png')
contar_e_desenhar_objetos(pathI, 'resultado_canny_padrao.png', 'contagem_metodo_canny.png')
contar_e_desenhar_objetos(pathI, 'image2_resultado_canny_highboost.png', 'contagem_metodo_combinado.png')