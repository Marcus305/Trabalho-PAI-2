from PIL import Image
import numpy as np
import os
import multiprocessing
from tqdm import tqdm

# --- Funções e Definições Globais ---
# Mantendo sua função original de filtro de mediana
def filtro_mediana(imagem, tamanho_janela=5):
    """Aplica um filtro de mediana na imagem."""
    offset = tamanho_janela // 2
    altura, largura = imagem.shape
    resultado = np.copy(imagem)

    for i in range(offset, altura - offset):
        for j in range(offset, largura - offset):
            vizinhos = imagem[i-offset:i+offset+1, j-offset:j+offset+1].flatten()
            resultado[i, j] = np.median(vizinhos)
    
    return resultado.astype(np.uint8)

# A função de processamento aceita uma tupla de argumentos para o multiprocessing
def processar_imagem(args):
    """
    Carrega uma imagem, aplica uma série de filtros (com loops manuais) e salva os resultados.
    """
    caminho_entrada, pasta_saida = args # Desempacota os argumentos
    try:
        # Extrai o nome do arquivo sem extensão
        base_name = os.path.basename(caminho_entrada)
        fileName = os.path.splitext(base_name)[0]

        # --- 1. Carregamento e pré-processamento da imagem ---
        with Image.open(caminho_entrada) as img:
            matriz_original = np.array(img.convert('L'), dtype=np.float32)
        
        altura, largura = matriz_original.shape

        # Calcula a média para ajustar os parâmetros
        media = np.mean(matriz_original)

        if media < 50:
            A = 10.0
            limiar_diferenca = 90
        elif media < 170:
            A = 2.0
            limiar_diferenca = 80
        else:
            A = 1.4
            limiar_diferenca = 9

        # # --- 2. Aplica suavização Gaussiana 3x3 (Loop Original) ---
        # kernel = np.array([[1, 4, 1], [4, 8, 4], [1, 4, 1]], dtype=np.float32) / 28.0
        # suavizada = np.zeros_like(matriz_original)
        # for i in range(1, altura - 1):
        #     for j in range(1, largura - 1):
        #         regiao = matriz_original[i-1:i+2, j-1:j+2]
        #         suavizada[i, j] = np.sum(regiao * kernel)

        # # --- 3. Filtro High-Boost (Lógica Original) ---
        # matriz_original_mediana = filtro_mediana(matriz_original)
        # janela_suavizacao = 10 if media < 50 else 9
        # suavizada_mediana = filtro_mediana(suavizada, tamanho_janela=janela_suavizacao)
        
        # highboost = (A * matriz_original_mediana) - suavizada_mediana
        # highboost = np.clip(highboost, 0, 255).astype(np.uint8)

        # --- 4. Detecção de bordas com Laplaciano (Loop Original) ---
        laplaciano = np.zeros((altura, largura), dtype=np.int32)
        mascara = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
        mascara2 = np.array([[0, -1, 0], [-1,  4, -1], [0, -1, 0]])
        for i in range(1, altura - 1):
            for j in range(1, largura - 1):
                regiao = matriz_original[i-1:i+2, j-1:j+2]
                laplaciano[i, j] = np.sum(regiao * mascara2)
        
        contorno = np.where(np.abs(laplaciano) > limiar_diferenca, 255, 0).astype(np.uint8)

        # --- 5. Salva os resultados na pasta de saída ---
        caminho_highboost = os.path.join(pasta_saida, f"{fileName}_highboost.png")
        caminho_contorno = os.path.join(pasta_saida, f"{fileName}_contorno.png")

        #Image.fromarray(highboost).save(caminho_highboost)
        Image.fromarray(contorno).save(caminho_contorno)
        
        return f"Sucesso: {base_name}"

    except Exception as e:
        return f"Erro em {os.path.basename(caminho_entrada)}: {e}"

# --- Bloco Principal de Execução com Processamento Paralelo ---
if __name__ == "__main__":
    # Garante que o multiprocessing funcione corretamente em diferentes SOs
    multiprocessing.freeze_support()

    pasta_input = "my_result"
    pasta_saida = "my_result_2_times"

    if not os.path.isdir(pasta_input):
        print(f"Erro: A pasta de entrada '{pasta_input}' não foi encontrada.")
    else:
        os.makedirs(pasta_saida, exist_ok=True)
        
        extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        
        # Cria uma lista de tarefas (cada tarefa é uma tupla de argumentos para a função)
        tarefas = [
            (os.path.join(pasta_input, nome_arquivo), pasta_saida)
            for nome_arquivo in os.listdir(pasta_input)
            if nome_arquivo.lower().endswith('_contorno.png')
        ]

        if not tarefas:
            print("Nenhum arquivo de imagem válido encontrado na pasta 'input'.")
        else:
            # Usa o número de núcleos de CPU disponíveis
            num_processos = os.cpu_count()
            print(f"Iniciando processamento paralelo com {num_processos} núcleos para {len(tarefas)} imagens.")

            # Cria um "pool" de processos para executar as tarefas em paralelo
            with multiprocessing.Pool(processes=num_processos) as pool:
                # 'pool.imap_unordered' processa as tarefas e retorna os resultados assim que ficam prontos
                # 'tqdm' cria a barra de progresso
                resultados = list(tqdm(pool.imap_unordered(processar_imagem, tarefas), total=len(tarefas)))
            
            print("-" * 50)
            print("Processamento em lote concluído!")
            # Opcional: imprimir o status de cada arquivo
            for res in resultados:
                if "Erro" in res:
                    print(res)