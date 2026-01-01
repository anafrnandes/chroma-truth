import os
import urllib.request
from PIL import Image
from tqdm import tqdm

REAL_FACES_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/9b/Andrea_Merkel_July_2010_-_3zu4.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/14/Albert_Einstein_1947.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c3/Ellen_DeGeneres_2011.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/1b/Tom_Cruise_by_Gage_Skidmore.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/8c/Cristiano_Ronaldo_2018.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/6f/Dwayne_Johnson_Hercules_2014_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/a9/Enzo_Ferrari_fotografato_da_Giuseppe_Enrie.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f6/Queen_Elizabeth_II_in_March_2015.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/George-W-Bush.jpeg",
    "https://upload.wikimedia.org/wikipedia/commons/2/22/Will_Smith_Cannes_2017.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/George_Clooney_66%C3%A8me_Festival_de_Venise_%28Mostra%29.jpg",
]


def download_manual_list():
    print("A baixar lista manual de faces reais...")

    # Caminho: chroma-truth/data/real
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'data', 'real')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Configurar User-Agent para não sermos bloqueados
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    count = 0
    for i, url in enumerate(tqdm(REAL_FACES_URLS)):
        try:
            # 1. Definir nome temporário
            temp_path = os.path.join(save_dir, f"temp_{i}.jpg")

            # 2. Baixar
            urllib.request.urlretrieve(url, temp_path)

            # 3. Abrir e Processar (Crop na cara seria ideal, mas Resize serve para teste)
            img = Image.open(temp_path).convert('RGB')
            img = img.resize((256, 256), Image.Resampling.LANCZOS)

            # 4. Guardar final
            final_path = os.path.join(save_dir, f"real_{i:04d}.jpg")
            img.save(final_path)

            # 5. Limpar temp
            os.remove(temp_path)
            count += 1

        except Exception as e:
            print(f"Erro ao baixar {url}: {e}")

    print(f"\nConcluído! Tens {count} imagens reais na pasta data/real.")
    print("DICA: Para o trabalho final, baixa mais fotos manualmente e põe nesta pasta.")


if __name__ == "__main__":
    download_manual_list()