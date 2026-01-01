import os
import numpy as np
from PIL import Image, ImageDraw

# Caminhos
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base_dir, 'data', 'test_samples', 'teste_astronauta.jpg')
save_path = os.path.join(base_dir, 'data', 'test_samples', 'fake_astronauta.jpg')

# Abrir imagem
img = Image.open(img_path).convert("RGB")

# Criar um "Fake" (Pintar um quadrado cinzento na cara)
draw = ImageDraw.Draw(img)
# Coordenadas (x0, y0, x1, y1) - ajusta se necessário para acertar na cara
draw.rectangle([200, 150, 300, 250], fill=(128, 128, 128))

img.save(save_path)
print(f"Deepfake criado em: {save_path}")