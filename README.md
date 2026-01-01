# Chroma Truth: Deepfake Detection via Chromatic Reconstruction 

**Trabalho Prático 2 para Redes Neuronais e Aprendizagem Profunda - Inteligência Artificial e Ciência de Dados**

**Universidade da Beira Interior (UBI)**

**Autora:** Ana Fernandes

**Data:** Janeiro 2026

---

## Sobre o Projeto
Este projeto investiga a hipótese **"Chroma Truth"**: a premissa de que a manipulação digital (Deepfakes) quebra a coerência física natural entre a luminância (luz) e a crominância (cor) de uma face.

Foi implementada uma rede **U-Net** em regime **Self-Supervised** que aprende a colorir faces reais a partir de inputs em escala de cinzentos. Durante a inferência, o erro de reconstrução cromática é utilizado para gerar Mapas de Calor (Heatmaps) e classificar a imagem como *Real* ou *Fake*.

### Resultados Principais
* **AUC Score:** 0.52 (O modelo comporta-se como um classificador aleatório para o dataset global).
* **Sucessos:** Deteta eficazmente manipulações de *splicing* e edições locais (ex: troca de boca/olhos).
* **Limitações:** O modelo demonstrou sensibilidade a iluminação complexa em imagens reais (falsos positivos) e dificuldade em detetar fakes gerados por Modelos de Difusão modernos.

---

## Estrutura do Repositório

```text
chroma-truth/
├── data/               # (Ignorado pelo Git) Imagens de treino e teste
├── models/             # Pesos do modelo treinado (.pth)
├── notebooks/          # Notebook Jupyter final com a demonstração
├── src/                # Código fonte
│   ├── dataset.py      # Processamento Lab e DataLoaders
│   ├── model.py        # Arquitetura U-Net
│   ├── train.py        # Script de treino
│   ├── detect.py       # Script de inferência e visualização (Heatmaps)
│   ├── evaluate.py     # Geração de métricas (ROC, Histogramas)
│   └── generate_fakes.py # Script para criar fakes por splicing
├── chroma_truth.pdf    # Relatório Científico Final (IEEE Format)
├── requirements.txt    # Dependências do Python
└── README.md           # Este ficheiro
```
---
## Como Correr o Projeto

### Instalação

```bash
git clone [https://github.com/anafrnandes/chroma-truth.git](https://github.com/anafrnandes/chroma-truth.git)
cd chroma-truth
pip install -r requirements.txt
```
### Treino (Opcional)

O modelo aprende a colorir faces reais.

```bash
python src/train.py
```
Nota: Requer imagens na pasta `data/real`.

### Deteção
Para testar numa imagem e ver o Heatmap:

```bash
python src/detect.py
```


### Avaliação Científica
Para gerar a Curva ROC e os Histogramas de erro:

```bash
python src/evaluate.py
```

---

## Download do Modelo
Devido ao limite de tamanho do GitHub (>100MB), o ficheiro de pesos chroma_unet.pth não está incluído neste repositório.

---

## Licença 

Projeto desenvolvido para fins académicos
