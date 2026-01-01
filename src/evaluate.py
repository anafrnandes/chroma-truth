import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from model import ColorUNet
from dataset import ChromaDataset
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_error(model, dataloader):
    errors = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            L = batch['L'].to(DEVICE)
            ab_real = batch['ab'].to(DEVICE)

            # Previsão
            ab_pred = model(L)

            # Calcular erro médio da imagem (MSE)
            # Fórmula E = ||ab - ab_orig||
            diff = (ab_pred - ab_real) ** 2
            error_map = torch.sqrt(torch.sum(diff, dim=1))  # Soma nos canais
            mean_error = error_map.mean(dim=(1, 2))  # Média espacial

            errors.extend(mean_error.cpu().numpy())
    return errors


def run_evaluation():
    print("A iniciar Avaliação Científica...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Carregar Modelo
    model = ColorUNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(base_dir, 'models', 'chroma_unet.pth'), map_location=DEVICE))

    # 2. Datasets
    real_path = os.path.join(base_dir, 'data', 'real')
    fake_path = os.path.join(base_dir, 'data', 'fake')

    dataset_real = ChromaDataset(real_path)
    dataset_fake = ChromaDataset(fake_path)

    loader_real = DataLoader(dataset_real, batch_size=1, shuffle=False)
    loader_fake = DataLoader(dataset_fake, batch_size=1, shuffle=False)

    print(f"A calcular erros para {len(dataset_real)} imagens Reais...")
    errors_real = calculate_error(model, loader_real)

    print(f"A calcular erros para {len(dataset_fake)} imagens Fake...")
    errors_fake = calculate_error(model, loader_fake)

    # 3. Visualização dos Histogramas
    plt.figure(figsize=(10, 5))
    plt.hist(errors_real, bins=20, alpha=0.7, label='Reais', color='green')
    plt.hist(errors_fake, bins=20, alpha=0.7, label='Fakes', color='red')
    plt.axvline(x=0.05, color='black', linestyle='--', label='Threshold (0.05)')
    plt.title("Distribuição do Erro de Reconstrução Cromática")
    plt.xlabel("Erro Médio (MSE)")
    plt.ylabel("Contagem")
    plt.legend()
    plt.savefig('histograma_erros.png')
    plt.show()

    # 4. Curva ROC e AUC (Obrigatório para o Paper!)
    # Labels: 0 para Real, 1 para Fake
    y_true = [0] * len(errors_real) + [1] * len(errors_fake)
    y_scores = errors_real + errors_fake

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('curva_roc.png')
    plt.show()

    print(f"AUC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    run_evaluation()