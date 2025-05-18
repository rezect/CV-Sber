import cv2
import numpy as np
import math
from ultralytics import YOLO
from scipy.optimize import minimize

MAX_FRAMES_HOME_APPEARENCE = 30

# TODO: детектить ВСЕ камни -> считать их реальные координаты -> учитывать только те, кто в пределах площадки

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
        self.house_radius_meters = 1.83  # Радиус дома
        self.hogline_distance_meters = 6.4  # Расстояние от центра дома до линии хог
        self.stone_tadius_meters = 0.279  # Расстояние от центра дома до линии хог

        self.hogline_to_house = self.hogline_distance_meters / \
            self.house_radius_meters  # На сколько hogline больше home
        self.stone_to_house = self.stone_tadius_meters / self.house_radius_meters  # Какую часть радиус камня составляет от радиуса дома

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
        house_radius = 1.83  # Радиус дома
        hog_line_y = 6.4     # Позиция линии хог
        field_length = 45.72  # Длина игрового поля
        field_width = 5.0     # Ширина игрового поля

        # Допуск для проверки нахождения точки на линии
        epsilon = 0.1

        # Проверка нахождения в доме
        distance_from_center = (x**2 + y**2)**0.5
        if distance_from_center <= house_radius:
            return "IN_HOUSE"

        # Проверка нахождения на линии хог
        if hog_line_y - y < epsilon:
            # Проверяем, что точка не выходит за ширину поля
            if abs(x) <= field_width / 2:
                return "ON_HOG_LINE"

        # Проверка нахождения на игровом поле
        if 0 <= y <= field_length and abs(x) <= field_width / 2:
            return "ON_PLAYING_FIELD"

        return "OUT_OF_FIELD"

    def get_real_coords(self, home_box, point, is_front_view):
        """
        Обрабатывает координаты на изображении в реальные координаты относительно центра дома

        Args:
            home_bos: координаты видимого дома [x1, y1, x2, y2]
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

        return cv2.perspectiveTransform(point_array, H)[0][0]
        
    def process_frame(self, frame):
        """
        Обрабатывает кадр для подсчета камней
        
        Args:
            frame: кадр видео
            
        Returns:
            словарь с результатами анализа
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        # Извлечение информации о боксах и классах
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

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

            prev_home_radius_y = self.last_known_home['radius'][1]
            prev_home_center_y = self.last_known_home['center'][1]

            if abs(home_radius_y - prev_home_radius_y) > 1.1 * prev_home_radius_y or abs(home_center_y - prev_home_center_y) > 10:
                # Если новый радиус или координата центра дома по оси y сильно отличаются, то это новый дом => ненадежный
                is_this_new_home = True
                is_this_new_home = False
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

            # Подготовка результатов
            results = {
                'stones_in_house': {
                    'total': len(stones_in_house),
                    'yellow': sum(1 for s in stones_in_house if s['color'] == 'yellow'),
                    'red': sum(1 for s in stones_in_house if s['color'] == 'red')
                },
                'stones_before_hogline': {
                    'total': len(stones_before_hogline),
                    'yellow': sum(1 for s in stones_before_hogline if s['color'] == 'yellow'),
                    'red': sum(1 for s in stones_before_hogline if s['color'] == 'red')
                },
                'home_center': home_center,
                'home_radius': home_radius
            }

            # Визуализация для отладки
            if self.debug:
                # Рисуем дом и линию хог
                if is_trusted_home:
                    cv2.ellipse(frame, (int(home_center[0]), int(home_center[1])),
                                (int(home_radius[0]), int(home_radius[1])),
                                0, 0, 360, (255, 0, 0), 2)

                    cv2.ellipse(frame, (int(home_center[0]), int(home_center[1])),
                                (int(home_radius[0] * self.hogline_to_house),
                                    int(home_radius[1] * self.hogline_to_house)),
                                0, 0, 360, (255, 0, 0), 2)

                # Рисуем камни
                for stone in stones_in_house:
                    color = (0, 255, 255) if stone['color'] == 'yellow' else (
                        0, 0, 255)
                    cv2.rectangle(frame,
                                (int(stone['box'][0]), int(stone['box'][1])),
                                (int(stone['box'][2]), int(stone['box'][3])),
                                color, 2)

                for stone in stones_before_hogline:
                    color = (0, 255, 255) if stone['color'] == 'yellow' else (
                        0, 0, 255)
                    cv2.rectangle(frame,
                                (int(stone['box'][0]), int(stone['box'][1])),
                                (int(stone['box'][2]), int(stone['box'][3])),
                                color, 2)

                cv2.imshow('Debug', frame)
                cv2.waitKey(1)

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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Подготовка вывода видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра
            frame_result = self.process_frame(frame)

            if frame_result:
                results.append(frame_result)

                # Визуализация результатов на кадре
                if output_path:
                    output_frame = frame.copy()
                    home_center = frame_result['home_center']
                    home_radius = frame_result['home_radius']

                    # Рисуем дом
                    cv2.ellipse(output_frame, (int(home_center[0]), int(home_center[1])),
                                (int(home_radius[0]), int(home_radius[1])),
                                0, 0, 360, (255, 0, 0), 2)

                    # Обновляем информацию о камнях
                    self.home_stones['yellow'] = frame_result['stones_in_house']['yellow']
                    self.home_stones['red'] = frame_result['stones_in_house']['red']
                    self.hogline_stones['yellow'] = frame_result['stones_before_hogline']['yellow']
                    self.hogline_stones['red'] = frame_result['stones_before_hogline']['red']

                    # Рисуем информацию о камнях
                    text_yellow = f"Yellow: House={frame_result['stones_in_house']['yellow']}, Hog={frame_result['stones_before_hogline']['yellow']}"
                    text_red = f"Red: House={frame_result['stones_in_house']['red']}, Hog={frame_result['stones_before_hogline']['red']}"

                    cv2.putText(output_frame, text_yellow, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)
                    cv2.putText(output_frame, text_red, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                    out.write(output_frame)
            else:
                # Если дом не найден, записываем кадр с имеющимися значениями
                if output_path:
                    output_frame = frame.copy()

                    # Рисуем информацию о камнях
                    text_yellow = f"Yellow: House={self.home_stones['yellow']}, Hog={self.hogline_stones['yellow']}"
                    text_red = f"Red: House={self.home_stones['red']}, Hog={self.hogline_stones['red']}"

                    cv2.putText(output_frame, text_yellow, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)
                    cv2.putText(output_frame, text_red, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    
                    out.write(output_frame)

            frame_idx += 1
            print(
                f"\r({frame_idx/frame_count*100:.1f}%)", end="")

        print("\nVideo processing completed!")
        cap.release()
        if output_path:
            out.release()

        if self.debug:
            cv2.destroyAllWindows()

        return results
    

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
