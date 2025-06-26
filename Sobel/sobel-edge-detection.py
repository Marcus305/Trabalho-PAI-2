# detect_sobel_ultra_optimized.py
import os
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def read_image_with_pillow(filename):
    try:
        with Image.open(filename) as img:
            grayscale_img = img.convert('L')
            width, height = grayscale_img.size
            pixel_data = list(grayscale_img.getdata())
            image_matrix = [pixel_data[i * width:(i + 1) * width] for i in range(height)]
            return image_matrix, width, height
    except Exception as e:
        print(f"Erro ao ler o arquivo {filename}: {e}")
        return None, 0, 0

def write_image_with_pillow(filename, image_matrix, width, height):
    try:
        img = Image.new('L', (width, height))
        pixel_data = [pixel for row in image_matrix for pixel in row]
        img.putdata(pixel_data)
        img.save(filename)
    except Exception as e:
        print(f"Erro ao salvar o arquivo {filename}: {e}")

def process_chunk(args):
    image, width, start_row, end_row = args
    sobel_chunk = [[0] * width for _ in range(end_row - start_row)]
    
    for r in range(max(1, start_row), min(end_row, len(image)-1)):
        for c in range(1, width-1):
            # Cálculo direto de Gx e Gy
            gx = (image[r-1][c+1] + 2*image[r][c+1] + image[r+1][c+1]) - \
                 (image[r-1][c-1] + 2*image[r][c-1] + image[r+1][c-1])
            
            gy = (image[r-1][c-1] + 2*image[r-1][c] + image[r-1][c+1]) - \
                 (image[r+1][c-1] + 2*image[r+1][c] + image[r+1][c+1])
            
            magnitude = abs(gx) + abs(gy)
            sobel_chunk[r - start_row][c] = magnitude
            
    return start_row, end_row, sobel_chunk

def apply_sobel_filter_ultra_optimized(image, width, height):
    num_workers = os.cpu_count() or 4
    chunk_size = (height + num_workers - 1) // num_workers
    chunks = []
    
    for i in range(num_workers):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, height)
        chunks.append((image, width, start_row, end_row))
    
    sobel_image = [[0] * width for _ in range(height)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_chunk, chunks)
        
        for start_row, end_row, chunk_data in results:
            for r in range(start_row, end_row):
                sobel_image[r] = chunk_data[r - start_row]
    
    # Normalização
    max_magnitude = max(max(row) for row in sobel_image)
    if max_magnitude > 0:
        for r in range(height):
            for c in range(width):
                sobel_image[r][c] = int((sobel_image[r][c] / max_magnitude) * 255)
    
    return sobel_image

if __name__ == "__main__":
    start_time = time.time()
    INPUT_FOLDER = "../input"
    SOBEL_OUTPUT_FOLDER = "sobel_edge_detection_output_ultra_optimized"
    os.makedirs(SOBEL_OUTPUT_FOLDER, exist_ok=True)

    print("="*50)
    print("SCRIPT ULTRA OTIMIZADO: DETECÇÃO DE BORDAS COM SOBEL")
    print(f"Pasta de Entrada: '{INPUT_FOLDER}'")
    print(f"Pasta de Saída: '{SOBEL_OUTPUT_FOLDER}'")
    print("="*50)

    if not os.path.isdir(INPUT_FOLDER):
        print(f"\nERRO: A pasta de entrada '{INPUT_FOLDER}' não foi encontrada.")
        print("Crie a pasta e coloque suas imagens nela.")
    else:
        files_to_process = os.listdir(INPUT_FOLDER)
        num_processed = 0
        for filename in files_to_process:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(INPUT_FOLDER, filename)
                print(f"\n--- Processando: {filename} ---")
                
                original_image, width, height = read_image_with_pillow(input_path)
                if original_image:
                    sobel_image = apply_sobel_filter_ultra_optimized(original_image, width, height)
                    
                    base_name, _ = os.path.splitext(filename)
                    output_path = os.path.join(SOBEL_OUTPUT_FOLDER, f"sobel_{base_name}.png")
                    
                    write_image_with_pillow(output_path, sobel_image, width, height)
                    print(f"Resultado do Sobel salvo em: {output_path}")
                    num_processed += 1
        
        end_time = time.time()
        print("\n" + "="*50)
        print("Detecção de bordas concluída!")
        print(f"Total de imagens processadas: {num_processed}")
        print(f"Tempo de execução: {end_time - start_time:.2f} segundos.")
        print("="*50)
