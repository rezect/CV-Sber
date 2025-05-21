import cv2
import csv
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Параметры модели
CONFIDENCE_TRASHOLD = 0.8

# Некоторые пути
OUTPUT_CSV_PATH = 'output.csv'
MODEL_PATH = "runs/detect/curling_detector/weights/best.pt"
VIDEO_PATH = "end.mov"
OUTPUT_PATH = "results/curling_results_new.mp4"  # Путь для результата (видео)


class CurlingStoneCounter:
    def __init__(self, model_path, conf):
        """
        Инициализация счетчика камней для кёрлинга

        Args:
            model_path: путь к модели YOLOv8
            conf: порог уверенности
        """

        self.model = YOLO(model_path)
        self.conf = conf

        # Параметры кёрлинга (в метрах)
        self.house_radius_meters = 1.83         # Радиус дома
        self.hogline_distance_meters = 6.4      # Расстояние от центра дома до линии хог
        self.stone_radius_meters = 0.292 / 2    # Радиу камня
        self.field_width = 4.75                 # Ширина площадки

    def estimate_full_circle(self, partial_box, frame_heigth):
        """
        Оценивает параметры полного круга по видимой части

        Args:
            partial_box: видимая часть дома [x1, y1, x2, y2]

        Returns:
            center_x, center_y, radius_x, radius_y
        """

        wigth = abs(partial_box[0] - partial_box[2])

        heigth = abs(partial_box[1] - partial_box[3])

        center_x = None
        center_y = None
        radius_x = None
        radius_y = None

        if frame_heigth - partial_box[3] < heigth / 20:
            # Вид сверху
            center_x = abs(partial_box[0] + partial_box[2]) / 2
            center_y = abs(2 * partial_box[1] + wigth) / 2
            radius_x = wigth / 2
            radius_y = radius_x
        else:
            # Вид спереди
            center_x = abs(partial_box[0] + partial_box[2]) / 2
            center_y = abs(partial_box[1] + partial_box[3]) / 2
            radius_x = wigth / 2
            radius_y = heigth / 2

        return center_x, center_y, radius_x, radius_y

    def is_point_on_playing_area(self, point):
        """
        Проверяет положение точки на игровом поле в кёрлинге.

        Параметры:
        - point: кортеж (x, y) с координатами точки в метрах относительно центра дома

        Возвращает:
        - "IN_HOUSE": если точка находится в доме (в радиусе 1.83 м от центра)
        - "ON_PLAYING_FIELD": если точка находится на игровом поле, но не в доме и до линии хог
        - "OUT_OF_FIELD": если точка за пределами игрового поля

        Координатная система:
        - Центр координат (0, 0) - центр дома
        - y - против направления бросания камней
        - x - вправо от y, если смотреть из центра в сторону y
        """

        x, y = point

        # Проверка нахождения в доме
        distance_from_center = (x**2 + y**2)**0.5
        if distance_from_center <= self.house_radius_meters:
            return "IN_HOUSE"

        # Проверка нахождения на игровом поле
        if 0 <= y <= self.hogline_distance_meters and abs(x) <= self.field_width / 2:
            return "ON_PLAYING_FIELD"

        return "OUT_OF_FIELD"

    def get_real_coords(self, home_box, point, is_front_view, stone_box):
        """
        Обрабатывает координаты на изображении в реальные координаты относительно центра дома

        Args:
            home_box: координаты дома [x1, y1, x2, y2]
            point: координаты точки [x, y]

        Returns:
            cords: координаты в СО относительно центра дома [x, y]
        """
        if is_front_view:
            # Если вид спереди, то надо немного подвинуть координаты по оси Y вперед (вверх от дома), тк имеется небольшое искажение
            home_x_diametr = home_box[2] - home_box[0]
            home_y_diametr = home_box[3] - home_box[1]
            cos_angle = home_y_diametr / home_x_diametr

            # Применяем смещение
            point = self.cords_bias(
                point, cos_angle, stone_box)

        x_center = (home_box[0] + home_box[2]) / 2
        y_center = (home_box[1] + home_box[3]) / 2

        if is_front_view:
            image_points = np.array([
                [x_center, home_box[3]],
                [x_center, home_box[1]],
                [home_box[2], y_center],
                [home_box[0], y_center]
            ], dtype=np.float32)
        else:
            image_points = np.array([
                [x_center, home_box[1]],
                [x_center, home_box[3]],
                [home_box[0], y_center],
                [home_box[2], y_center]
            ], dtype=np.float32)

        home_radius = self.house_radius_meters

        real_points = np.array([
            [0, home_radius],    # Top
            [0, -home_radius],   # Bottom
            [-home_radius, 0],   # Left
            [home_radius, 0]     # Right
        ], dtype=np.float32)

        H, _ = cv2.findHomography(image_points, real_points)

        point_array = np.array([[point]], dtype=np.float32)

        real_point_coords = cv2.perspectiveTransform(point_array, H)[0][0]

        return real_point_coords

    def cords_bias(self, point, cos_b, stone_box):
        """
        Немного сдвигает точку центра камня, если вид на поле под углом

        Args:
            point: координаты точки [x, y]
            cos_b: косинус угла наклона плоскости камеры к полю
            stone_box: координаты области камня [x1, y1, x2, y2]

        Returns:
            point: новая точка центра камня на изображении
        """
        # Данные о камне
        scale = abs(stone_box[0] - stone_box[2]) / \
            (self.stone_radius_meters * 2)
        im_h = abs(stone_box[1] - stone_box[3])
        a = self.stone_radius_meters * scale

        # Доп переменные
        y_top = stone_box[1]
        y_bottom = stone_box[3]

        # Получаем отношение реальной координаты центра на изображении к имеющейся
        l = im_h / cos_b
        rat = (a) / (l / 2)

        # Получаем длину (в пикселях) до реального центра на изображении от нижней границы
        c = abs(y_top - y_bottom) / 2
        x = rat * c

        new_y = y_bottom - x

        return (point[0], new_y)

    def process_frame(self, frame, frame_idx):
        """
        Обрабатывает кадр для подсчета камней

        Args:
            frame: кадр видео

        Returns:
            словарь с результатами анализа
        """
        # Получаем область интереса (ROI)
        height, width = frame.shape[:2]
        square_size = height

        if width < height:
            square_size = width

        x_center = width // 2
        y_center = height // 2
        x_start = max(0, x_center - square_size // 2)
        y_start = max(0, y_center - square_size // 2)
        x_end = min(width, x_center + square_size // 2)
        y_end = min(height, y_center + square_size // 2)

        roi = frame[y_start:y_end, x_start:x_end]

        # Получаем предсказания модели по области интереса
        model_pred = self.model.predict(
            roi, conf=self.conf, verbose=False)[0]

        # Обработка результатов детекции
        boxes = model_pred.boxes.xyxy.cpu().numpy()
        classes = model_pred.boxes.cls.cpu().numpy()

        # Преобразуем координаты ROI обратно к оригинальному кадру
        for i in range(len(boxes)):
            boxes[i][0] += x_start  # x_min
            boxes[i][1] += y_start  # y_min
            boxes[i][2] += x_start  # x_max
            boxes[i][3] += y_start  # y_max

        # Раздляем дома и камни
        home_boxes = boxes[classes == 0] if len(boxes) > 0 else np.array([])
        yellow_stones = boxes[classes == 1] if len(boxes) > 0 else np.array([])
        red_stones = boxes[classes == 2] if len(boxes) > 0 else np.array([])

        # Найти параметры дома
        home_center = None
        home_radius = None

        # Флаг - вид спереди или нет. Нужен, тк тогда прийдется отзеркалить координаты камней
        is_front_view = False

        if len(home_boxes) > 0:
            # Берем самый большой детектированный дом
            home_box = home_boxes[np.argmax((home_boxes[:, 2] - home_boxes[:, 0]) *
                                            (home_boxes[:, 3] - home_boxes[:, 1]))]

            # Оцениваем полный круг дома
            home_center_x, home_center_y, home_radius_x, home_radius_y = self.estimate_full_circle(
                home_box, frame.shape[0])
            home_center = (home_center_x, home_center_y)
            home_radius = (home_radius_x, home_radius_y)

            # По координатам центра и длинам радиуса получаем координаты дома вида [x1, y1, ,x2, y2]
            home_box = np.array([
                home_center_x - home_radius_x,
                home_center_y - home_radius_y,
                home_center_x + home_radius_x,
                home_center_y + home_radius_y
            ])

            if abs(home_radius_x - home_radius_y) > 20:
                is_front_view = True

        # Подсчет камней в доме и до линии хог
        stones_in_house = []
        stones_before_hogline = []
        invalid_stones = []

        # Объединяем все камни
        all_stones = np.vstack([yellow_stones, red_stones]) if len(yellow_stones) > 0 and len(red_stones) > 0 else (
            yellow_stones if len(yellow_stones) > 0 else (red_stones if len(red_stones) > 0 else np.array([])))

        # Если у нас есть информация о доме, подсчитываем камни
        if home_center and home_radius:
            for stone in all_stones:
                image_stone_center = self.get_stone_center(stone)

                real_stone_center = self.get_real_coords(
                    home_box, image_stone_center, is_front_view, stone)

                point_position = self.is_point_on_playing_area(
                    real_stone_center)

                # Проверяем, где находится камень
                if point_position == "IN_HOUSE":
                    stones_in_house.append({
                        'box': stone,
                        'image_coords': image_stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red',
                        'real_coords': real_stone_center
                    })
                elif point_position == "ON_PLAYING_FIELD" or point_position == "ON_HOG_LINE":
                    stones_before_hogline.append({
                        'box': stone,
                        'image_coords': image_stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red',
                        'real_coords': real_stone_center
                    })
                else:
                    invalid_stones.append({
                        'box': stone,
                        'image_coords': image_stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red'
                    })
                    continue

                # Записываем в ксв файл результаты
                with open(OUTPUT_CSV_PATH, mode="a", newline="") as file:
                    csv_writer = csv.writer(file)
                    team = "yellow" if stone in yellow_stones else "red"
                    x_cm, y_cm = real_stone_center
                    csv_writer.writerow([frame_idx, team, x_cm, y_cm])

        else:
            # Дом не найден или он ненадежный
            for stone in all_stones:
                image_stone_center = self.get_stone_center(stone)
                invalid_stones.append({
                    'box': stone,
                    'image_coords': image_stone_center,
                    'color': 'yellow' if stone in yellow_stones else 'red'
                })

        # Подготовка результатов
        results = {
            'stones_in_house': {
                'stones': stones_in_house,
                'total': len(stones_in_house),
                'yellow': sum(1 for s in stones_in_house if s['color'] == 'yellow'),
                'red': sum(1 for s in stones_in_house if s['color'] == 'red')
            },
            'stones_before_hogline': {
                'stones': stones_before_hogline,
                'total': len(stones_before_hogline),
                'yellow': sum(1 for s in stones_before_hogline if s['color'] == 'yellow'),
                'red': sum(1 for s in stones_before_hogline if s['color'] == 'red')
            },
            'invalid_stones': invalid_stones,
            'home_center': home_center,
            'home_radius': home_radius
        }

        return results

    def get_stone_center(self, stone_box):
        """Возвращает центр камня"""
        return ((stone_box[0] + stone_box[2]) / 2, (stone_box[1] + stone_box[3]) / 2)

    def process_video(self, video_path, output_path=None):
        """
        Обрабатывает видео и подсчитывает камни

        Args:
            video_path: путь к видео
            output_path: путь для сохранения результатов видео (если None - не сохраняем)

        Returns:
            список результатов анализа для каждого кадра
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Пишем первую строчку в csv
        with open(OUTPUT_CSV_PATH, mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["frame_id", "team", "x_cm", "y_cm"])
            file.close()

        # Характеристики текущего кадра
        img_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vis_height = img_heigth
        scale = vis_height / \
            (self.hogline_distance_meters + self.house_radius_meters)
        vis_width = self.field_width * scale
        total_heigth = img_heigth
        total_width = int(img_width + vis_width)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (total_width, total_heigth))

        results = []
        frame_idx = 0

        # Обрабатываем видео покадрово
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Обрабатываем кадр
            frame_result = self.process_frame(frame, frame_idx)

            # Добавляем в список если обработка прошла успешно
            if frame_result:
                results.append(frame_result)

            visualization = self.add_visualisation(frame, frame_result)

            # if visualization is not None:
            #     cv2.imshow('Real Coordinate Validation', visualization)
            #     key = cv2.waitKey(0)
            #     if key == ord('q'):
            #         break

            if writer:
                writer.write(visualization)

            frame_idx += 1
            print(
                f"\r({frame_idx/frame_count*100:.1f}%)", end="")

        print("\nVideo processing completed!")
        cap.release()
        if output_path:
            writer.release()

        cv2.destroyAllWindows()

        return results

    def add_visualisation(self, frame, frame_result):
        """
        Добавляет визуализацию и боксы к найденным камням

        Args:
            frame: original video frame
            frame_result: results from process_frame

        Returns:
            visualization: image with real coordinate visualization
        """
        real_vis_height = frame.shape[0]

        # Переменная для превращения метров в пиксели
        scale = real_vis_height / \
            (self.hogline_distance_meters + self.house_radius_meters)

        # Получаем ширину поля в пикселях для визуализации
        real_vis_width = int(self.field_width * scale)
        real_vis = np.ones((real_vis_height, real_vis_width, 3),
                           dtype=np.uint8) * 255

        # Вычисляем координаты начала СО
        center_point = (real_vis_width // 2, int(real_vis_height -
                        self.house_radius_meters * scale))

        # Рисуем круги дома
        for r in [self.house_radius_meters, 1.22,
                  0.61, 0.025]:
            radius_px = int(r * scale)
            cv2.circle(real_vis, center_point, radius_px, (200, 200, 200), 2)

        # Рисусем координатные оси
        cv2.line(real_vis, (center_point[0], 0), (center_point[0], real_vis_height),
                 (0, 0, 0), 1)
        cv2.line(real_vis, (0, center_point[1]), (real_vis_width, center_point[1]),
                 (0, 0, 0), 1)
        cv2.putText(real_vis, "Real-world coordinates (meters)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Делаем копию кадра с визуализацией
        img_vis = frame.copy()

        # Если дома на кадре нет, просто рисуем всё пустое
        if not frame_result:
            img_vis_resized = cv2.resize(
                img_vis, (frame.shape[1], real_vis_height))
            combined_vis = np.hstack((img_vis_resized, real_vis))
            return combined_vis

        # Получаем инфу о доме
        home_center = frame_result['home_center']
        home_radius = frame_result['home_radius']

        # Рисуем дом
        if home_center and home_radius:
            cv2.ellipse(img_vis, (int(home_center[0]), int(home_center[1])),
                        (int(home_radius[0]), int(home_radius[1])),
                        0, 0, 360, (255, 0, 0), 2)

        # Рисуем все камни
        self.draw_stones(frame_result['stones_in_house']['stones'],
                         img_vis, real_vis, center_point, scale, False)

        self.draw_stones(frame_result['stones_before_hogline']['stones'],
                         img_vis, real_vis, center_point, scale, False)

        self.draw_stones(frame_result['invalid_stones'],
                         img_vis, real_vis, center_point, scale, True)

        # Add counts
        cv2.putText(img_vis, f"Yellow in house: {frame_result['stones_in_house']['yellow']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_vis, f"Red in house: {frame_result['stones_in_house']['red']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_vis, f"Yellow before hogline: {frame_result['stones_before_hogline']['yellow']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_vis, f"Red before hogline: {frame_result['stones_before_hogline']['red']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Combine both visualizations side by side
        img_vis_resized = cv2.resize(
            img_vis, (frame.shape[1], real_vis_height))
        combined_vis = np.hstack((img_vis_resized, real_vis))

        return combined_vis

    def draw_stones(self, all_stones, img_vis, real_vis, center_point, scale, is_invalid_stones):
        for stone in all_stones:
            color = (0, 255, 255) if stone['color'] == 'yellow' else (
                0, 0, 255)
            if is_invalid_stones:
                color = (200, 200, 200)
            # Рисуем камень на видео
            cv2.rectangle(img_vis,
                          (int(stone['box'][0]), int(stone['box'][1])),
                          (int(stone['box'][2]), int(stone['box'][3])),
                          color, 2)
            # Рисуем камень на симуляции

            if not is_invalid_stones:
                point_x = int(center_point[0] +
                              stone['real_coords'][0] * scale)
                point_y = int(center_point[1] -
                              stone['real_coords'][1] * scale)
                cv2.circle(real_vis, (point_x, point_y),
                           int(self.stone_radius_meters * scale), color, -1)

                # Добавляем подпись с координатами
                coord_text = f"({stone['real_coords'][0]:.2f}, {stone['real_coords'][1]:.2f})"
                cv2.putText(real_vis, coord_text, (point_x + 5, point_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    # Создаем счетчик камней
    counter = CurlingStoneCounter(MODEL_PATH, CONFIDENCE_TRASHOLD)

    # Обрабатываем видео
    results = counter.process_video(VIDEO_PATH, OUTPUT_PATH)

    # Вывод статистики
    print("\nСтатистика по видео:")

    if results:
        # Максимальное количество камней каждого цвета в доме
        max_yellow_in_house = max(
            r['stones_in_house']['yellow'] for r in results)
        max_red_in_house = max(r['stones_in_house']['red'] for r in results)

        print(
            f"Максимальное количество желтых камней в доме: {max_yellow_in_house}")
        print(
            f"Максимальное количество красных камней в доме: {max_red_in_house}")
        print(
            f"Максимальное общее количество камней в доме: {max(r['stones_in_house']['total'] for r in results)}")
    else:
        print("Не удалось обнаружить дом на протяжении всего видео.")
