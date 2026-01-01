import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, color, transform
from PIL import Image


class ChromaDataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=(256, 256)):
        """
        Args:
            root_dir (string): Caminho para a pasta com as imagens.
            mode (string): 'train' ou 'val'
            image_size (tuple): Tamanho para o qual vamos redimensionar (H, W).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


        if len(self.image_files) == 0:
            print(f"Nenhuma imagem encontrada em {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Carregar a imagem
        img_name = os.path.join(self.root_dir, self.image_files[idx])


        image = Image.open(img_name).convert('RGB')
        image = np.array(image)

        # 2. Redimensionar (Resize)
        # O skimage espera float entre 0 e 1 quando redimensiona
        image = transform.resize(image, self.image_size, anti_aliasing=True)

        # 3. Converter RGB -> LAB
        # L range: [0, 100], a/b range: aprox [-128, 128]
        lab_image = color.rgb2lab(image)

        # Separar canais
        l_channel = lab_image[:, :, 0]  # Canal L (Luminância)
        ab_channels = lab_image[:, :, 1:]  # Canais ab (Crominância)

        # 4. Normalização (Crucial para a rede convergir!)

        # Normalizar L para o intervalo [-1, 1] (estava 0 a 100)
        l_channel = (l_channel / 50.0) - 1.0

        # Normalizar ab para o intervalo [-1, 1] (estava -128 a 128)
        ab_channels = ab_channels / 128.0

        # 5. Converter para Tensores do PyTorch
        # O PyTorch quer canais primeiro: (C, H, W) em vez de (H, W, C)
        l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()  # Adiciona dimensão do canal -> (1, 256, 256)
        ab_tensor = torch.from_numpy(ab_channels.transpose((2, 0, 1))).float()  # -> (2, 256, 256)

        return {'L': l_tensor, 'ab': ab_tensor, 'path': img_name}


if __name__ == "__main__":
    print("A testar o Dataset...")

    # 1. Descobre onde está este ficheiro (pasta src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Sobe um nível para ir para a raiz do projeto (chroma-truth)
    project_root = os.path.dirname(current_dir)
    # 3. Junta o caminho para a pasta de testes
    test_path = os.path.join(project_root, 'data', 'test_samples')

    print(f"A procurar imagens em: {test_path}")

    # Verifica se a pasta existe antes de tentar carregar
    if os.path.exists(test_path):
        dataset = ChromaDataset(root_dir=test_path)

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Imagem carregada.")
            print(f"Shape L (Input): {sample['L'].shape}")
            print(f"Shape ab (Target): {sample['ab'].shape}")
        else:
            print("pasta existe, mas está vazia! Põe lá uma imagem (jpg/png).")
    else:
        print(f"Erro: A pasta não existe. Cria a pasta: {test_path}")