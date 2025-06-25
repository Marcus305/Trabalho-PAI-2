import cv2
import numpy as np
from PIL import Image
import os
import multiprocessing
from tqdm import tqdm

def aplicar_canny(args):
    """
    Carrega uma imagem, aplica o detector de bordas Canny e salva o resultado.
    Aceita uma tupla de argumentos para compatibilidade com multiprocessing.
    """
    caminho_entrada, pasta_saida = args  # Desempacota os argumentos
    
    try:
        # Extrai o nome base do arquivo para criar o nome do arquivo de saída
        base_name = os.path.basename(caminho_entrada)
        # Transforma 'image2_highboost.png' em 'image2_canny.png'
        fileName_output = base_name.replace('_highboost.png', '_canny.png')

        # Carrega a imagem_highboost
        with Image.open(caminho_entrada) as img_pil:
            img_cv = np.array(img_pil.convert('L'))

        # Aplica um desfoque Gaussiano antes do Canny para reduzir ruído
        # (5, 5) é o tamanho do kernel de desfoque. Pode ser ajustado.
        img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        
        # Aplica o Canny
        # threshold1 e threshold2 são os limiares mínimo e máximo.
        # Ajustar esses valores é a principal forma de refinar o resultado do Canny.
        bordas_canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        # Salva o resultado
        caminho_saida = os.path.join(pasta_saida, fileName_output)
        Image.fromarray(bordas_canny).save(caminho_saida)
        
        return f"Sucesso: {base_name}"

    except Exception as e:
        return f"Erro em {os.path.basename(caminho_entrada)}: {e}"

# --- Bloco Principal de Execução com Processamento Paralelo ---
if __name__ == "__main__":
    # Garante que o multiprocessing funcione corretamente
    multiprocessing.freeze_support()

    # Define a pasta de onde leremos os arquivos _highboost.png
    pasta_leitura = "input"
    # Define a pasta onde salvaremos os resultados do Canny
    pasta_saida = "canny"

    # Verifica se a pasta de leitura existe
    if not os.path.isdir(pasta_leitura):
        print(f"Erro: A pasta de leitura '{pasta_leitura}' não foi encontrada.")
        print("Certifique-se de executar o script anterior primeiro para gerar os arquivos '_highboost.png'.")
    else:
        # Cria a pasta de saída se ela não existir
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Cria a lista de tarefas, processando apenas os arquivos corretos
        tarefas = [
            (os.path.join(pasta_leitura, nome_arquivo), pasta_saida)
            for nome_arquivo in os.listdir(pasta_leitura)
            #if nome_arquivo.lower().endswith('_highboost.png') # Filtro importante!
        ]

        if not tarefas:
            print(f"Nenhum arquivo '_highboost.png' encontrado na pasta '{pasta_leitura}'.")
        else:
            # Usa o número de núcleos de CPU disponíveis
            num_processos = os.cpu_count()
            print(f"Iniciando detector Canny com {num_processos} núcleos para {len(tarefas)} imagens.")

            # Cria um "pool" de processos para executar as tarefas em paralelo
            with multiprocessing.Pool(processes=num_processos) as pool:
                # `imap_unordered` processa as tarefas e `tqdm` cria a barra de progresso
                resultados = list(tqdm(pool.imap_unordered(aplicar_canny, tarefas), total=len(tarefas)))
            
            print("-" * 50)
            print("Processamento Canny concluído!")
            # Opcional: imprimir o status de cada arquivo
            for res in resultados:
                if "Erro" in res:
                    print(res)