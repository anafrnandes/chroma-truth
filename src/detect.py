import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from model import ColorUNet
from dataset import ChromaDataset

# CONFIGURAÇÕES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (256, 256)
THRESHOLD = 0.05  # Valor de corte. Se o erro for maior que isto -> É FAKE!


def lab_to_rgb(L, ab):
    """Converte Tensor Lab de volta para Imagem RGB para visualização."""
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    lab_image = np.zeros((256, 256, 3))
    lab_image[:, :, 0] = L.cpu().numpy()
    lab_image[:, :, 1:] = ab.cpu().numpy().transpose(1, 2, 0)
    return color.lab2rgb(lab_image)


def calculate_error_map(ab_pred, ab_real):
    """Calcula a diferença pixel-a-pixel (Mapa de Calor)."""
    diff = (ab_pred - ab_real) ** 2
    # Raiz quadrada da soma dos erros dos canais a e b
    error_map = torch.sqrt(torch.sum(diff, dim=0))
    return error_map


def detect_fake():
    print(f"A iniciar 'Chroma Truth' em {DEVICE}...")

    # 1. Preparar caminhos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_samples')
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'chroma_unet.pth')

    # 2. Carregar Modelo
    if not os.path.exists(model_path):
        print("Erro: Modelo não encontrado. Corre o train.py primeiro!")
        return

    model = ColorUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Modelo carregado.")

    # 3. Carregar Imagens
    dataset = ChromaDataset(root_dir=data_path, image_size=IMAGE_SIZE)
    if len(dataset) == 0:
        print("A pasta data/test_samples está vazia.")
        return

    # Vamos analisar a PRIMEIRA imagem da pasta
    data = dataset[0]
    img_path = data['path']
    print(f"A analisar imagem: {os.path.basename(img_path)}")

    with torch.no_grad():
        L = data['L'].unsqueeze(0).to(DEVICE)  # Input
        ab_real = data['ab'].to(DEVICE)  # Ground Truth

        # A. A rede pinta a imagem
        ab_pred = model(L)

        # B. Calculamos o erro
        # Removemos dimensão de batch
        error_map = calculate_error_map(ab_pred.squeeze(0), ab_real)

        # C. Score Final (Média do erro na imagem toda)
        mean_error = error_map.mean().item()

    # 4. Classificação (Real vs Fake)
    classification = "FAKE" if mean_error > THRESHOLD else "REAL"
    color_text = 'red' if classification == "FAKE" else 'green'

    print(f"Erro Médio: {mean_error:.4f}")
    print(f"Classificação: {classification}")

    # 5. Visualização Final
    L_img = L.squeeze().squeeze().cpu().numpy()
    rgb_reconstructed = lab_to_rgb(L.squeeze(0).squeeze(0), ab_pred.squeeze(0))

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Resultado: {classification} (Erro: {mean_error:.4f})",
                 fontsize=16, color=color_text, weight='bold')

    # O que a rede viu (Preto e Branco)
    plt.subplot(1, 3, 1)
    plt.title("Input (Canal L)")
    plt.imshow(L_img, cmap='gray')
    plt.axis('off')

    # Como a rede pintou (Reconstrução)
    plt.subplot(1, 3, 2)
    plt.title("Reconstrução Cromática")
    plt.imshow(rgb_reconstructed)
    plt.axis('off')

    # O Mapa de Calor (O "Detector")
    plt.subplot(1, 3, 3)
    plt.title("Mapa de Erro (Heatmap)")
    plt.imshow(error_map.cpu().numpy(), cmap='jet', vmin=0, vmax=0.5)  # vmax controla o contraste
    plt.colorbar(label="Erro de Cor")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    detect_fake()