import cv2
import numpy as np
import math
from ultralytics import YOLO
from scipy.optimize import minimize

MAX_FRAMES_HOME_APPEARENCE = 100

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

        self.last_known_home = None
        self.home_persistence = 0
        self.home_missing_frames = 0

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
    
    def process_frame(self, frame, frame_heigth):
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

        if len(home_boxes) > 0:
            # TODO: Сделать не самый большой, а самый центральный
            # Берем самый большой детектированный дом
            home_box = home_boxes[np.argmax((home_boxes[:, 2] - home_boxes[:, 0]) *
                                            (home_boxes[:, 3] - home_boxes[:, 1]))]

            # Оцениваем полный круг
            home_center_x, home_center_y, home_radius_x, home_radius_y = self.estimate_full_circle(
                home_box, frame.shape[0])
            home_center = (home_center_x, home_center_y)
            home_radius = (home_radius_x, home_radius_y)

            is_this_new_home = False

            if self.last_known_home:
                if self.last_known_home['radius'][1] > 1.2 * home_radius_y:
                    is_this_new_home = True

            # Обновляем последние известные параметры дома
            self.last_known_home = {
                'center': home_center,
                'radius': home_radius
            }

            # Если новый дом, то скидываем счетчики
            if is_this_new_home:
                self.home_persistence = 0
                self.home_frames_missing = 0
            else:
                self.home_frames_missing = 0
                self.home_persistence += 1
                
        else:
            self.home_frames_missing = 0
            self.home_persistence = 0

        # Если у нас есть информация о доме, подсчитываем камни
        if home_center and home_radius:
            # Объединяем все камни
            all_stones = np.vstack([yellow_stones, red_stones]) if len(yellow_stones) > 0 and len(red_stones) > 0 else (
                yellow_stones if len(yellow_stones) > 0 else (red_stones if len(red_stones) > 0 else np.array([])))

            # Подсчет камней в доме и на линии хог
            stones_in_house = []
            stones_on_hogline = []

            # Пиксельное соотношение между радиусом дома и расстоянием до линии хог
            pixels_per_meter = home_radius[0] / self.house_radius_meters
            hogline_distance_pixels = self.hogline_distance_meters * pixels_per_meter

            for stone in all_stones:
                stone_center = self.get_stone_center(stone)

                # Проверяем, где находится камень
                # TODO: Преобразовывать всё таки в координаты (понадобится далее)
                # Берем круг, плюс половина диаметра шара (тк касающиеся линии тоже считаются в доме)
                if self.is_point_in_ellipse(stone_center, home_center, home_radius[0] * (1 + self.stone_to_house / 2), home_radius[1] * (1 + self.stone_to_house / 2)):
                    stones_in_house.append({
                        'box': stone,
                        'center': stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red'
                    })
                elif self.is_point_in_ellipse(stone_center, home_center, home_radius[0] * self.hogline_to_house, home_radius[1] * self.hogline_to_house):
                    stones_on_hogline.append({
                        'box': stone,
                        'center': stone_center,
                        'color': 'yellow' if stone in yellow_stones else 'red'
                    })

            # Подготовка результатов
            if self.home_persistence > MAX_FRAMES_HOME_APPEARENCE and home_radius[0] == home_radius[1]:
                results = {
                    'stones_in_house': {
                        'total': len(stones_in_house),
                        'yellow': sum(1 for s in stones_in_house if s['color'] == 'yellow'),
                        'red': sum(1 for s in stones_in_house if s['color'] == 'red')
                    },
                    'stones_on_hogline': {
                        'total': len(stones_on_hogline),
                        'yellow': sum(1 for s in stones_on_hogline if s['color'] == 'yellow'),
                        'red': sum(1 for s in stones_on_hogline if s['color'] == 'red')
                    },
                    'home_center': home_center,
                    'home_radius': home_radius,
                    'is_home_valid': self.home_persistence > 50
                }
            else:
                return None
            
            # Визуализация для отладки
            if self.debug:
                # Рисуем дом
                cv2.ellipse(frame, (int(home_center[0]), int(home_center[1])),
                            (int(home_radius[0]), int(home_radius[1])),
                            0, 0, 360, (255, 0, 0), 2)

                # Рисуем линию хог
                cv2.circle(frame, (int(home_center[0]), int(home_center[1])),
                           int(hogline_distance_pixels), (0, 255, 0), 2)

                # Рисуем камни
                for stone in stones_in_house:
                    color = (0, 255, 255) if stone['color'] == 'yellow' else (
                        0, 0, 255)
                    cv2.rectangle(frame,
                                  (int(stone['box'][0]), int(stone['box'][1])),
                                  (int(stone['box'][2]), int(stone['box'][3])),
                                  color, 2)

                for stone in stones_on_hogline:
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
            # Дом не найден и нет последних известных параметров
            return None

    def get_stone_center(self, stone_box):
        """Возвращает центр камня"""
        return ((stone_box[0] + stone_box[2]) / 2, (stone_box[1] + stone_box[3]) / 2)
    
    def is_point_in_ellipse(self, point, center, radius_x, radius_y):
        """Проверяет, находится ли точка внутри овала (эллипса)"""
        dx = point[0] - center[0]
        dy = point[1] - center[1]

        return (dx/radius_x)**2 + (dy/radius_y)**2 <= 1

    def is_point_in_circle(self, point, center, radius):
        """Проверяет, находится ли точка внутри круга"""
        return math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2) <= radius
    
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
            frame_result = self.process_frame(frame, height)

            if frame_result:
                results.append(frame_result)

                # Визуализация результатов на кадре
                if output_path:
                    output_frame = frame.copy()
                    is_home_valid = frame_result['is_home_valid']
                    home_center = frame_result['home_center']
                    home_radius = frame_result['home_radius']

                    # Рисуем дом
                    cv2.ellipse(output_frame, (int(home_center[0]), int(home_center[1])),
                                (int(home_radius[0]), int(home_radius[1])),
                                0, 0, 360, (255, 0, 0), 2)

                    # Рисуем информацию о камнях
                    text_yellow = f"Yellow: House={frame_result['stones_in_house']['yellow']}, Hog={frame_result['stones_on_hogline']['yellow']}"
                    text_red = f"Red: House={frame_result['stones_in_house']['red']}, Hog={frame_result['stones_on_hogline']['red']}"

                    cv2.putText(output_frame, text_yellow, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)
                    cv2.putText(output_frame, text_red, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                    out.write(output_frame)
            else:
                # Если дом не найден, записываем оригинальный кадр
                if output_path:
                    out.write(frame)

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
