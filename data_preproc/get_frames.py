# Обрезает видео и режет видео на кадры

import cv2
import os
from tqdm import tqdm

# Обрезаем видео до квадрата
def crop_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    size = min(width, height)  # Берём меньшую сторону
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (size, size))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y_start:y_start+size, x_start:x_start+size]
        out.write(cropped)
    
    cap.release()
    out.release()

# Вытаскиваем по кадру из видео в папку frames/images
def extract_frames(video_path, interval=30):
    """
    Извлекает кадры из видео с указанным интервалом
    
    Args:
        video_path (str): Путь к видеофайлу
        interval (int): Интервал между извлекаемыми кадрами
        
    Returns:
        list: Список путей к сохраненным кадрам
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Всего кадров: {total_frames}, FPS: {fps}")
    
    frame_paths = []
    
    # Извлекаем кадры с заданным интервалом
    for frame_idx in tqdm(range(0, total_frames, interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_path = f"frames/images/frame_{frame_idx:06d}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    cap.release()
    print(f"Извлечено {len(frame_paths)} кадров")
    return frame_paths


if __name__ == "__main__":
    os.makedirs('frames', exist_ok=True)
    os.makedirs('frames/images', exist_ok=True)
    os.makedirs('frames/labels', exist_ok=True)

    video_path = 'croped_end.mov'

    crop_video("end.mov", video_path)

    frames = extract_frames(video_path, interval=30)
