import os
import re

def renomear_arquivos_de_forma_inteligente():
    """
    Renomeia apenas os arquivos que ainda não estão no formato 'image<numero>'.
    Ele detecta o maior número já usado e continua a partir dele.
    """
    pasta_alvo = 'input'
    
    if not os.path.isdir(pasta_alvo):
        print(f"Erro: A pasta '{pasta_alvo}' não foi encontrada.")
        print("Por favor, crie a pasta 'input' e coloque seus arquivos dentro dela.")
        return

    try:
        nomes_dos_arquivos = sorted(os.listdir(pasta_alvo))
    except OSError as e:
        print(f"Erro ao acessar a pasta '{pasta_alvo}': {e}")
        return

    # 1. Separar arquivos que já foram renomeados dos que precisam ser
    arquivos_para_renomear = []
    maior_numero_existente = 0
    
    # Regex para encontrar o padrão 'image' seguido de um ou mais dígitos
    # Ex: image1.jpg, image125.png
    regex_imagem = re.compile(r'^image(\d+)\..+$', re.IGNORECASE)

    for nome_arquivo in nomes_dos_arquivos:
        # Processar apenas arquivos, ignorar pastas
        caminho_completo = os.path.join(pasta_alvo, nome_arquivo)
        if not os.path.isfile(caminho_completo):
            continue

        match = regex_imagem.match(nome_arquivo)
        if match:
            # O arquivo já está no formato correto. Vamos extrair o número.
            numero_atual = int(match.group(1))
            if numero_atual > maior_numero_existente:
                maior_numero_existente = numero_atual
        else:
            # O arquivo tem um nome diferente (ex: 'ash-_-ismail...jpg')
            # e precisa ser renomeado.
            arquivos_para_renomear.append(nome_arquivo)

    # 2. Se não há arquivos novos, não há nada a fazer
    if not arquivos_para_renomear:
        print("Nenhum arquivo novo para renomear. Tudo já está organizado.")
        return

    # 3. Iniciar o contador a partir do próximo número disponível
    contador = maior_numero_existente + 1
    
    print(f"Encontrados {len(arquivos_para_renomear)} arquivo(s) para renomear. Começando a partir de image{contador}...")

    # 4. Renomear apenas os arquivos necessários
    for nome_antigo in arquivos_para_renomear:
        caminho_antigo = os.path.join(pasta_alvo, nome_antigo)
        _ , extensao = os.path.splitext(nome_antigo)

        novo_nome = f"image{contador}{extensao}"
        caminho_novo = os.path.join(pasta_alvo, novo_nome)

        try:
            os.rename(caminho_antigo, caminho_novo)
            print(f"'{nome_antigo}'  --->  '{novo_nome}'")
            contador += 1
        except OSError as e:
            print(f"Não foi possível renomear '{nome_antigo}'. Erro: {e}")

    print("\nProcesso de renomeação concluído com sucesso!")

# Executa a função principal quando o script é rodado
if __name__ == "__main__":
    renomear_arquivos_de_forma_inteligente()