import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Barra de progresso bonita

# Importar os nossos módulos
from dataset import ChromaDataset
from model import ColorUNet

# CONFIGURAÇÕES
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100  # Para testar rápido. No projeto real, talvez 50 ou 100.
IMAGE_SIZE = (256, 256)


def train_model():
    print(f"A iniciar treino em: {DEVICE}")

    # 1. Preparar Dados
    # Truque do caminho para encontrar a pasta data/test_samples automaticamente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'real')

    dataset = ChromaDataset(root_dir=data_path, image_size=IMAGE_SIZE)

    if len(dataset) == 0:
        print("Erro: Não há imagens para treinar")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Inicializar Modelo, Loss e Optimizer
    model = ColorUNet().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Loop de Treino
    model.train()  # Colocar o modelo em modo de treino

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loop):
            # Buscar dados e enviar para a GPU/CPU
            L = batch['L'].to(DEVICE)  # Input (Preto e Branco)
            ab = batch['ab'].to(DEVICE)  # Target (Cor Real)

            # Forward Pass (A rede tenta adivinhar)
            preds = model(L)

            # Calcular o erro (Quão longe a cor prevista está da real?)
            loss = criterion(preds, ab)

            # Backward Pass (Aprender com o erro)
            optimizer.zero_grad()  # Limpar gradientes antigos
            loss.backward()  # Calcular novos gradientes
            optimizer.step()  # Atualizar pesos da rede

            # Atualizar barra de progresso
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # Média do erro na época
        print(f"Epoch {epoch + 1} finalizada. Loss Média: {epoch_loss / len(dataloader):.6f}")

    # 4. Guardar o Modelo
    # Criar pasta models se não existir
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)

    save_path = os.path.join(models_dir, 'chroma_unet.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado com sucesso em: {save_path}")


if __name__ == "__main__":
    train_model()