from PIL import Image
import os

# Diretório onde estão as imagens
image_dir = 'dataset/training'

# Inicializando variáveis
unique_sizes = set()
total_width = 0
total_height = 0
image_count = 0

# Percorrer todas as pastas e arquivos no diretório
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Verifica extensões comuns de imagens
            image_path = os.path.join(root, file)
            with Image.open(image_path) as img:
                size = img.size  # (largura, altura)
                unique_sizes.add(size)
                total_width += size[0]
                total_height += size[1]
                image_count += 1

# Exibir os tamanhos únicos encontrados
print("Tamanhos únicos de imagens encontradas:")
for size in unique_sizes:
    print(size)

# Calcular e exibir a média
if image_count > 0:
    avg_width = total_width / image_count
    avg_height = total_height / image_count
    print(f"\nMédia das dimensões das imagens:")
    print(f"Largura média: {avg_width:.2f}")
    print(f"Altura média: {avg_height:.2f}")
else:
    print("Nenhuma imagem encontrada.")