import cv2
import numpy as np
import math
from ultralytics import YOLO
from scipy.optimize import minimize

MAX_FRAMES_HOME_APPEARENCE = 50

class CurlingStoneCounter:
    def __init__(self, model_path, conf, debug=False):
        """
        Инициализация счетчика камней для кёрлинга
        
        Args:
            model_path: путь к модели YOLOv8
            conf: порог уверенности
            debug: если True, будут отображаться визуализации для отладки
        """

        self.model = YOLO(model_path)
        self.conf = conf
        self.debug = debug

        # Параметры кёрлинга (в метрах)
        self.house_radius_meters = 1.83         # Радиус дома
        self.hogline_distance_meters = 6.4      # Расстояние от центра дома до линии хог
        self.stone_radius_meters = 0.292 / 2    # Радиу камня
        self.stone_heigth = 0.114               # Высота камня
        self.field_width = 4.75                 # Ширина площадки

        # TODO: Реализовать поумнее подсчет камней в доме и на линии хог
        self.hogline_stones = {
            'yellow': 0,
            'red': 0
        }
        self.home_stones = {
            'yellow': 0,
            'red': 0
        }

        self.last_known_home = {
            'center': (0, 0),
            'radius': (0, 0)
        }
        self.home_persistence = 0

        self.tracked_stones = []

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
        - "ON_HOG_LINE": если точка находится на линии хог (y = 6.4 м)
        - "ON_PLAYING_FIELD": если точка находится на игровом поле, но не в доме и не на линии хог
        - "OUT_OF_FIELD": если точка за пределами игрового поля
        
        Координатная система:
        - Центр координат (0, 0) - центр дома
        - y - против направления бросания камней
        - x - вправо от y, если смотреть из центра в сторону y
        """

        x, y = point

        # Параметры поля в кёрлинге (в метрах)
        field_length = 45.72  # Длина игрового поля

        # Допуск для проверки нахождения точки на линии
        epsilon = 0.1

        # Проверка нахождения в доме
        distance_from_center = (x**2 + y**2)**0.5
        if distance_from_center <= self.house_radius_meters:
            return "IN_HOUSE"

        # Проверка нахождения на линии хог
        if self.hogline_distance_meters - y < epsilon:
            # Проверяем, что точка не выходит за ширину поля
            if abs(x) <= self.field_width / 2:
                return "ON_HOG_LINE"

        # Проверка нахождения на игровом поле
        if 0 <= y <= field_length and abs(x) <= self.field_width / 2:
            return "ON_PLAYING_FIELD"

        return "OUT_OF_FIELD"

    def get_real_coords(self, home_box, point, is_front_view):
        """
        Обрабатывает координаты на изображении в реальные координаты относительно центра дома

        Args:
            home_box: координаты видимого дома [x1, y1, x2, y2]
            point: координаты точки [x, y]

        Returns:
            cords: координаты в СО относительно центра дома[x, y]
        """
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

        # if is_front_view:
        #     # Если вид спереди, то надо немного подвинуть координаты по оси Y вперед (вверх от дома), тк имеется небольшое искажение
        #     # Получаем косинус угла наклона
        #     home_x_diametr = home_box[2] - home_box[0]
        #     home_y_diametr = home_box[3] - home_box[1]
        #     cos_angle = home_y_diametr / home_x_diametr

        #     # Применяем смещение
        #     real_point_coords = self.cords_bias(
        #         real_point_coords, cos_angle)

        return real_point_coords
    
    # def cords_bias(self, point, cos_b):
    #     def ctg(x):
    #         return math.cos(x) / math.sin(x)
    #     # Данные о камне
    #     a = self.stone_radius_meters
    #     h = self.stone_heigth

    #     # Получаем b
    #     b = h * ctg(math.acos(cos_b))
    #     l = 2 * a + b
    #     rat = (l / 2) / (a)

        
    def process_frame(self, frame):
        """
        Обрабатывает кадр для подсчета камней
        
        Args:
            frame: кадр видео
            
        Returns:
            словарь с результатами анализа
        """
        model_pred = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        # Извлечение информации о боксах и классах
        boxes = model_pred.boxes.xyxy.cpu().numpy()
        classes = model_pred.boxes.cls.cpu().numpy()

        # Обработка результатов детекции
        home_boxes = boxes[classes == 0] if len(boxes) > 0 else np.array([])
        yellow_stones = boxes[classes == 1] if len(boxes) > 0 else np.array([])
        red_stones = boxes[classes == 2] if len(boxes) > 0 else np.array([])

        # Найти параметры дома
        home_center = None
        home_radius = None
        
        # Если есть дом, то новый ли он
        is_this_new_home = True
        is_trusted_home = False
        is_front_view = False

        if len(home_boxes) > 0:
            # Берем самый большой детектированный дом
            home_box = home_boxes[np.argmax((home_boxes[:, 2] - home_boxes[:, 0]) *
                                            (home_boxes[:, 3] - home_boxes[:, 1]))]

            # Оцениваем полный круг
            home_center_x, home_center_y, home_radius_x, home_radius_y = self.estimate_full_circle(
                home_box, frame.shape[0])
            home_center = (home_center_x, home_center_y)
            home_radius = (home_radius_x, home_radius_y)

            # Меняем home_box, пишем реальные корды
            home_box = np.array([
                home_center_x - home_radius_x,
                home_center_y - home_radius_y,
                home_center_x + home_radius_x,
                home_center_y + home_radius_y
            ])

            prev_home_radius_y = self.last_known_home['radius'][1]
            prev_home_center_y = self.last_known_home['center'][1]

            if abs(home_radius_y - prev_home_radius_y) > 1.1 * prev_home_radius_y or abs(home_center_y - prev_home_center_y) > 10:
                # Если новый радиус или координата центра дома по оси y сильно отличаются, то это новый дом => ненадежный
                is_this_new_home = True
                is_trusted_home = False
            else:
                is_this_new_home = False
                # Проверяем - надежный ли дом
                if self.home_persistence > MAX_FRAMES_HOME_APPEARENCE:
                    is_trusted_home = True

            if abs(home_radius_x - home_radius_y) > 20:
                is_front_view = True

            # Обновляем последние известные параметры дома
            self.last_known_home = {
                'center': home_center,
                'radius': home_radius
            }

            # Если новый дом, то скидываем счетчик
            if is_this_new_home:
                self.home_persistence = 0
            else:
                self.home_persistence += 1
        else:
            # Если дома нет на кадре, тоже скидываем счетчик
            self.home_persistence = 0
                
        # Если у нас есть информация о доме и он НАДЕЖНЫЙ, подсчитываем камни
        if home_center and home_radius and is_trusted_home:
            # Объединяем все камни
            all_stones = np.vstack([yellow_stones, red_stones]) if len(yellow_stones) > 0 and len(red_stones) > 0 else (
                yellow_stones if len(yellow_stones) > 0 else (red_stones if len(red_stones) > 0 else np.array([])))

            # Подсчет камней в доме и на линии хог
            stones_in_house = []
            stones_before_hogline = []

            for stone in all_stones:
                image_stone_center = self.get_stone_center(stone)

                real_stone_center = self.get_real_coords(
                    home_box, image_stone_center, is_front_view)
                
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
                elif point_position == "ON_PLAYING_FIELD":
                    stones_before_hogline.append({
                        'box': stone,
                        'image_coords': image_stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red',
                        'real_coords': real_stone_center
                    })

                self.home_stones = {
                    'yellow': sum(1 for s in stones_in_house if s['color'] == 'yellow'),
                    'red': sum(1 for s in stones_in_house if s['color'] == 'red')
                }

                self.hogline_stones = {
                    'yellow': sum(1 for s in stones_before_hogline if s['color'] == 'yellow'),
                    'red': sum(1 for s in stones_before_hogline if s['color'] == 'red')
                }

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
                'home_center': home_center,
                'home_radius': home_radius
            }

            return results
            
        else:
            # Дом не найден или он ненадежный
            return None

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

        # Характеристики нового видео
        img_heigth = 800
        img_width = 800
        vis_height = 800
        scale = vis_height / (self.hogline_distance_meters + self.house_radius_meters)
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame_result = self.process_frame(frame)

            # Add result to the list if it's valid
            if frame_result:
                results.append(frame_result)

            visualization = self.add_visualisation(frame, frame_result)

            if visualization is not None:
                if self.debug:
                    cv2.imshow('Real Coordinate Validation', visualization)
                    key = cv2.waitKey(0 if self.debug else 1)
                    if key == ord('q'):
                        break

                if writer:
                    writer.write(visualization)

            frame_idx += 1
            print(
                f"\r({frame_idx/frame_count*100:.1f}%)", end="")

        print("\nVideo processing completed!")
        cap.release()
        if output_path:
            writer.release()

        if self.debug:
            cv2.destroyAllWindows()

        return results


    def add_visualisation(self, frame, frame_result):
        """
        Adds a visualization of stones in real coordinates
    
        Args:
            frame: original video frame
            frame_result: results from process_frame
        
        Returns:
            visualization: image with real coordinate visualization
        """
        if not frame_result:
            return None
        
        # Extract home information
        home_center = frame_result['home_center']
        home_radius = frame_result['home_radius']

        # Create a white canvas for real-world visualization
        real_vis_height = 800

        # Переменная для превращения метров в пиксели
        scale = real_vis_height / (self.hogline_distance_meters + self.house_radius_meters)

        # Получаем ширину поля в пикселях для визуализации
        real_vis_width = int(self.field_width * scale)
        real_vis = np.ones((real_vis_height, real_vis_width, 3),
                        dtype=np.uint8) * 255

        # Вычисляем координаты начала СО
        center_point = (real_vis_width // 2, int(real_vis_height -
                        self.house_radius_meters * scale))
        
        # Draw concentric circles for house
        for r in [self.house_radius_meters, 1.22,
                  0.61, 0.025]:
            radius_px = int(r * scale)
            cv2.circle(real_vis, center_point, radius_px, (200, 200, 200), 2)
        
        # Draw coordinate axes
        cv2.line(real_vis, (center_point[0], 0), (center_point[0], real_vis_height),
                (0, 0, 0), 1)
        cv2.line(real_vis, (0, center_point[1]), (real_vis_width, center_point[1]),
                (0, 0, 0), 1)
        cv2.putText(real_vis, "Real-world coordinates (meters)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Create a visualization of original frame with detected objects
        img_vis = frame.copy()
        
        # Рисуем дом
        cv2.ellipse(img_vis, (int(home_center[0]), int(home_center[1])),
                    (int(home_radius[0]), int(home_radius[1])),
                    0, 0, 360, (255, 0, 0), 2)
        
        # Рисуем все камни
        for stone in frame_result['stones_in_house']['stones']:
            color = (0, 255, 255) if stone['color'] == 'yellow' else (
                0, 0, 255)
            # Рисуем камень на видео
            cv2.rectangle(img_vis,
                        (int(stone['box'][0]), int(stone['box'][1])),
                        (int(stone['box'][2]), int(stone['box'][3])),
                        color, 2)
            # Рисуем камень на симуляции
            point_x = int(center_point[0] + stone['real_coords'][0] * scale)
            point_y = int(center_point[1] - stone['real_coords'][1] * scale)
            cv2.circle(real_vis, (point_x, point_y),
                       int(self.stone_radius_meters * scale), color, -1)
            
        for stone in frame_result['stones_before_hogline']['stones']:
            color = (0, 255, 255) if stone['color'] == 'yellow' else (
                0, 0, 255)
            # Рисуем камень на видео
            cv2.rectangle(img_vis,
                        (int(stone['box'][0]), int(stone['box'][1])),
                        (int(stone['box'][2]), int(stone['box'][3])),
                        color, 2)
            # Рисуем камень на симуляции
            point_x = int(center_point[0] + stone['real_coords'][0] * scale)
            point_y = int(center_point[1] - stone['real_coords'][1] * scale)
            cv2.circle(real_vis, (point_x, point_y),
                       int(self.stone_radius_meters * scale), color, -1)
            
        # Add counts
        cv2.putText(img_vis, f"Yellow in house: {self.home_stones['yellow']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_vis, f"Red in house: {self.home_stones['red']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_vis, f"Yellow on hogline: {self.hogline_stones['yellow']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_vis, f"Red on hogline: {self.hogline_stones['red']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine both visualizations side by side
        img_vis_resized = cv2.resize(
            img_vis, (real_vis_height, real_vis_height))
        combined_vis = np.hstack((img_vis_resized, real_vis))

        return combined_vis

if __name__ == "__main__":
    # Пути к файлам
    model_path = "runs/detect/curling_detector/weights/best.pt"  # Путь к модели
    video_path = "croped_end.mov"   # Путь к видео
    output_path = "results/curling_results.mp4"  # Путь для результата

    # Создаем счетчик камней
    counter = CurlingStoneCounter(model_path, 0.3, debug=True)

    # Обрабатываем видео
    results = counter.process_video(video_path, output_path)

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
