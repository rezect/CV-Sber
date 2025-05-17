# Разделяет данные на тренировочные и валидационные

import os
import shutil

# Разделяем данные на тренировочные и валидационные
base_dir = 'frames'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

image_files = [f.replace('.jpg', '') for f in os.listdir(images_dir) if f.endswith('.jpg')]

# TODO: make image_files randomly shuffeled

val_count = int(len(image_files) * 0.1)

val_files = image_files[:val_count]
train_files = image_files[val_count:]

def copy_files(files, dest_dir):
    for file in files:
        shutil.copy2(
            os.path.join(images_dir, file + '.jpg'),
            os.path.join(dest_dir, 'images', file + '.jpg')
        )
        label_file = file + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy2(
                os.path.join(labels_dir, label_file),
                os.path.join(dest_dir, 'labels', label_file)
            )


if __name__ == "__main__":
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)

    print(f"Разделение завершено: {len(train_files)} файлов в train, {len(val_files)} файлов в val")