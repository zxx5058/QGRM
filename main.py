# coding:utf-8
import numpy as np
import pyqtgraph as pg
import torch
import sys
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
import time
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from Rutils.featuretrack import FeaturePointSelectionStrategy, ImproveOpticalFlow3
from Rutils.chooseface import choose_face_strategy, calculate_chest_abdomen_location
from Rutils.breath_utils import *

'''
This project is designed for fully automatic monitoring of human respiration rate. Please ensure that there is only one subject in the camera frame, seated upright, with the face fully visible during operation.
·count_frame: The index of the current frame being processed
·chest_abdomen_coords: Coordinates of the chest and abdomen regions
·ResWave: List used to store waveform data
·latest_breath_rate: The most recently calculated respiration rate
·k: The average distance of tracked feature points from the center of the image
'''


def weighted_breath_rate(current_rate, previous_rate, alpha):
    if previous_rate is None:
        return current_rate
    return alpha * current_rate + (1 - alpha) * previous_rate


def load_model(weights='best.pt', source='0', img_size=(640, 640)):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img_size, s=stride)  # 检查图像大小
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    fps = dataset.fps[0]

    return dataset, fps, device, model


class TrackingLostError(Exception):
    def __init__(self, message="FeaturePointsTrackingLost"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"TrackingLostError: {self.message}"


class VideoDisplay(QWidget):
    def __init__(self, fps, dataset, out, device, model, run_time, parent=None):
        super(VideoDisplay, self).__init__(parent)
        self.fps = fps
        self.dataset = dataset
        self.out = out
        self.device = device
        self.model = model
        self.run_time = run_time

        self.setWindowTitle("Respiratory Signal")
        self.frame_width = 600
        self.frame_height = 400

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.frame_width, self.frame_height)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setFixedSize(self.frame_width, self.frame_height)

        axis_pen = pg.mkPen(color=(0, 0, 0), width=3)  # 轴颜色和粗细
        font = QFont()
        font.setPointSize(16)
        tick_text_pen = pg.mkPen(color=(0, 0, 0))

        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setTickFont(font)
        self.plot_widget.getAxis('left').setTickFont(font)

        self.window_size = 300
        self.sk_values = np.zeros(self.window_size)
        self.timestamps = np.zeros(self.window_size)

        curve_pen = pg.mkPen(color=(0, 119, 182), width=5)
        self.curve = self.plot_widget.plot(self.timestamps, self.sk_values, pen=curve_pen)

        self.breath_rate_label = QLabel('Breath Rate: N/A', self)
        self.breath_rate_label.setAlignment(Qt.AlignCenter)
        self.breath_rate_label.setFixedHeight(80)
        self.breath_rate_label.setStyleSheet("font-size: 40px; font-weight: bold;")

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.image_label)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.setSizes([self.frame_width, self.frame_width])

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.breath_rate_label)
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)

        self.initUI()

    def initUI(self):
        self.processor = BreathingSignalProcessor(self.fps, self.dataset, self.out, self.device, self.model, self.run_time)
        self.processor.frame_signal.connect(self.update_image)
        self.processor.sk_signal.connect(self.update_sk)
        self.processor.breath_rate.connect(self.update_breath_rate)

        self.processor.start()

    def update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaling_factor = 0.3
        scaled_pixmap = pixmap.scaled(
            int(width * scaling_factor),
            int(height * scaling_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def update_sk(self, sk_value, timestamp):
        if len(self.sk_values) >= self.window_size:
            self.sk_values = np.roll(self.sk_values, -1)
            self.sk_values[-1] = sk_value
            self.timestamps = np.roll(self.timestamps, -1)
            self.timestamps[-1] = timestamp
        else:
            self.sk_values[len(self.sk_values)] = sk_value
            self.timestamps[len(self.timestamps)] = timestamp

        self.curve.setData(self.timestamps, self.sk_values)

    def update_breath_rate(self, breath_rate):
        self.breath_rate_label.setText(f'Breath Rate: {breath_rate:.2f}')


class BreathingSignalProcessor(QThread):
    frame_signal = Signal(np.ndarray)
    sk_signal = Signal(float, float)
    breath_rate = Signal(float)

    def __init__(self, fps, dataset, out, device, model, run_time):
        super(BreathingSignalProcessor, self).__init__()
        self.cumulative_vertical_movement = 0
        self.cumulative_threshold = 10
        self.prev_bbox = None
        self.fps = fps
        self.ResWave = []
        self.count_frame = 0
        self.high_cut = 1.3
        self.cache_len = 62
        self.calc_win = 20
        self.order = 61
        self.pre_breath_rate = 0
        self.filter_cache = [0 for _ in range(self.cache_len)]
        self.ppg_cache = [0 for _ in range(self.cache_len)]
        self.chest_abdomen_coords = None
        self.FPMap = None
        self.Image_gray = None
        self.latest_breath_rate = 0
        self.is_running = True
        self.dataset = dataset
        self.out = out
        self.final_res = []
        self.refresh_interval = 60
        self.device = device
        self.model = model
        self.prev_time = time.time()
        self.run_time = run_time
        self.start_time = None
        self.frame_count_for_fps = 0
        self.fps_start_time = time.time()
        self.Sk_buffer = []
        self.last_Sk = 0

    def run(self):
        self.start_time = time.time()
        self.prev_time = time.time()
        while self.is_running:
            for path, img, im0s, vid_cap, s in self.dataset:
                if not self.is_running:
                    return

                elapsed_time = time.time() - self.start_time
                print("Frame_Time:", elapsed_time)
                if elapsed_time > self.run_time:
                    print("Total_Frame:", self.count_frame)
                    out.release()
                    cv2.destroyAllWindows()
                    self.stop()
                    return

                frame = im0s[0].copy()
                self.frame_signal.emit(frame)

                if self.count_frame % (self.refresh_interval * self.fps) == 0 or self.FPMap is None or self.chest_abdomen_coords is None:
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.model.fp16 else img.float()
                    img /= 255.0

                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    pred = model(img, augment=False, visualize=False)
                    pred = non_max_suppression(pred, 0.6, 0.45, classes=[0], agnostic=False)

                    for i, det in enumerate(pred):
                        im0 = frame.copy()
                        if len(det):
                            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                            chosen_face = choose_face_strategy(det, im0)
                            if chosen_face:
                                self.chest_abdomen_coords = calculate_chest_abdomen_location(chosen_face, im0.shape)
                                chest_abdomen_x1, chest_abdomen_y1, chest_abdomen_x2, chest_abdomen_y2 = self.chest_abdomen_coords
                                cv2.rectangle(im0, (chest_abdomen_x1, chest_abdomen_y1),
                                              (chest_abdomen_x2, chest_abdomen_y2), (0, 255, 0), 2)
                                Image_ROI = im0[chest_abdomen_y1:chest_abdomen_y2, chest_abdomen_x1:chest_abdomen_x2]
                                self.FPMap, self.Image_gray, Sk = FeaturePointSelectionStrategy(Image_ROI)
                                break

                if self.chest_abdomen_coords is not None:
                    chest_abdomen_x1, chest_abdomen_y1, chest_abdomen_x2, chest_abdomen_y2 = self.chest_abdomen_coords
                    cv2.rectangle(frame, (chest_abdomen_x1, chest_abdomen_y1), (chest_abdomen_x2, chest_abdomen_y2),
                                  (0, 255, 0), 2)
                    self.out.write(frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    Image_ROI = frame[chest_abdomen_y1:chest_abdomen_y2, chest_abdomen_x1:chest_abdomen_x2]

                    if self.FPMap is not None:
                        try:
                            self.FPMap, self.Image_gray, Sk, is_tracking_lost, cumulative_vertical_movement = ImproveOpticalFlow3(Image_ROI, self.Image_gray, self.FPMap, Sk)
                            self.cumulative_vertical_movement += cumulative_vertical_movement
                            if is_tracking_lost:
                                self.chest_abdomen_coords = None
                                continue
                        except TrackingLostError:
                            self.chest_abdomen_coords = None
                            continue
                    self.ResWave.append(Sk)

                    b, a = generate_fir_lowpass(self.high_cut, self.fps, self.order)
                    for j in range(self.cache_len - 1):
                        self.filter_cache[j] = self.filter_cache[j + 1]
                        self.ppg_cache[j] = self.ppg_cache[j + 1]
                    self.ppg_cache[-1] = Sk

                    if self.count_frame <= self.order:
                        Sk_filtered = 0
                    else:
                        self.filter_cache = data_filter(self.filter_cache, b, a, self.ppg_cache)
                        if self.filter_cache[-2] == 0:
                            Sk_filtered = 0
                        else:
                            if np.abs(Sk_filtered) < 2:
                                Sk_filtered = self.filter_cache[-1] - self.filter_cache[-2]
                                if np.abs(Sk_filtered) > 1:
                                    Sk_filtered = np.sign(Sk_filtered) * 1
                            else:
                                Sk_filtered = 0
                    self.final_res.append(Sk_filtered)

                    self.sk_signal.emit(Sk_filtered, elapsed_time)
                    if self.count_frame % self.fps == 0 and self.count_frame > 12 * self.fps:
                        start = 0 if self.count_frame < self.calc_win * self.fps else self.count_frame - self.calc_win * self.fps
                        filtered_array = self.final_res[int(start):int(self.count_frame)]
                        peaks = find_and_select_peaks(filtered_array, int(self.fps * 0.6))

                        if peaks is None:
                            self.latest_breath_rate = self.pre_breath_rate
                        else:
                            self.latest_breath_rate = calculate_breath_rate_by_peaks(peaks, self.fps)
                            self.pre_breath_rate = self.latest_breath_rate
                    print("*"*20, self.latest_breath_rate)

                    self.breath_rate.emit(self.latest_breath_rate)

                    self.count_frame += 1
                    self.frame_count_for_fps += 1
                    if time.time() - self.fps_start_time >= 1.0:
                        real_time_fps = self.frame_count_for_fps / (time.time() - self.fps_start_time)
                        print("*"*60, real_time_fps)
                        self.frame_count_for_fps = 0
                        self.fps_start_time = time.time()

                    current_time = time.time()
                    time_interval = current_time - self.prev_time
                    self.prev_time = current_time
                    print("Time interval between Sk_filtered values: {:.3f} seconds".format(time_interval))

                    if len(self.Sk_buffer) < 5:
                        self.Sk_buffer.append(time_interval)
                    else:
                        median_value = np.median(self.Sk_buffer)
                        if time_interval > 2 * median_value:
                            print("Alert: Sk is more than twice the median value!")
                            self.Sk_buffer.pop(0)  # Remove the oldest Sk
                            self.Sk_buffer.append(time_interval)  # Add the current Sk
        self.stop()

    def stop(self):
        self.is_running = False
        self.quit()
        self.out.release()
        app.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = r'output.mp4'
    width = 640
    height = 480
    dataset, fps, device, model = load_model(weights='best.pt', source='0', img_size=(640, 640))
    print("FPS:", fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    display = VideoDisplay(fps, dataset, out, device, model, run_time=200)
    display.show()
    sys.exit(app.exec_())



