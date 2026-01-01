import os
import random
from PIL import Image, ImageFilter


def create_fakes():
    print("A iniciar o Gerador de Fakes...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_dir = os.path.join(base_dir, 'data', 'real')
    fake_dir = os.path.join(base_dir, 'data', 'fake')

    if not os.path.exists(fake_dir): os.makedirs(fake_dir)
    real_images = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]

    if len(real_images) < 2:
        print("Precisa de mais imagens em data/real!")
        return

    print(f"A gerar 10 fakes a partir de {len(real_images)} imagens reais...")

    for i in range(10):
        img_name_A = random.choice(real_images)  # Vítima
        img_name_B = random.choice(real_images)  # Doador

        img_A = Image.open(os.path.join(real_dir, img_name_A)).convert('RGB').resize((256, 256))
        img_B = Image.open(os.path.join(real_dir, img_name_B)).convert('RGB').resize((256, 256))

        # Máscara para cortar olhos ou boca
        mask = Image.new("L", (256, 256), 0)
        # Random: ou olhos ou boca
        if random.random() > 0.5:
            box = (60, 80, 196, 130)  # Olhos
        else:
            box = (80, 160, 176, 210)  # Boca

        mask.paste(255, box)
        mask = mask.filter(ImageFilter.GaussianBlur(10))  # Suavizar bordas

        # Colar B em A
        img_fake = Image.composite(img_B, img_A, mask)
        img_fake.save(os.path.join(fake_dir, f"fake_{i}.jpg"))

    print(f"Fakes criados em: {fake_dir}")


if __name__ == "__main__":
    create_fakes()