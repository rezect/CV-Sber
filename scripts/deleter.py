# Удаляет неразмеченные кадры из датасета

import os

images_dir = 'frames/images'
labels_dir = 'frames/labels'

# Получаем имена файлов без расширения из labels_dir
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

# Проходим по всем .jpg файлам в images_dir
for image_file in os.listdir(images_dir):
    if image_file.endswith('.jpg'):
        image_name = os.path.splitext(image_file)[0]
        if image_name not in label_files:
            image_path = os.path.join(images_dir, image_file)
            os.remove(image_path)
            print(f"Удалён файл: {image_path}")
