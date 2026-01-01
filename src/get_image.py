import os
from skimage import data
from PIL import Image

# Define o caminho
save_path = '../data/test_samples'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Carrega uma imagem de exemplo (Astronauta)
img = data.astronaut()
pil_img = Image.fromarray(img)

# Guarda na pasta
full_path = os.path.join(save_path, 'teste_astronauta.jpg')
pil_img.save(full_path)

print(f"Imagem criada em: {full_path}")