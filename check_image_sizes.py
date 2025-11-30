from PIL import Image
import os

image_dir = 'outputs/gold_samples/images'
if os.path.exists(image_dir):
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:10]
    for f in files:
        try:
            img = Image.open(os.path.join(image_dir, f))
            print(f'{f}: {img.size[0]}×{img.size[1]}')
            img.close()
        except Exception as e:
            print(f'{f}: ERROR - {e}')
else:
    print('No image directory found')
