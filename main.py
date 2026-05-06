import sys
import cv2
import torch
import time
from ultralytics import YOLO
from collections import defaultdict, deque

from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtMultimedia import QSoundEffect

MODEL_PATH = "yolov8n-pose.pt"

HISTORY_LEN = 5
FINISH_SCORE = 20

MIN_RISE = 3
MIN_JUMP = 5
MIN_FALL = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)


class CaptureWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Capture Window")

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:black;")

        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(self.capture_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_capture)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(50)

    def update_preview(self):
        if self.parent.current_frame is None:
            return

        frame = cv2.flip(self.parent.current_frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def capture_image(self):
        if self.parent.current_frame is None:
            return

        filename = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, self.parent.current_frame)
        print("Saved:", filename)


class JumpApp(QWidget):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # COUNTDOWN
        self.countdown = None
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)

        # TIMER PER PLAYER
        self.left_start_time = None
        self.right_start_time = None
        self.left_time = 0
        self.right_time = 0

        # SOUND
        self.snd_go = QSoundEffect()
        self.snd_jump = QSoundEffect()
        self.snd_finish = QSoundEffect()

        self.snd_go.setSource(QUrl.fromLocalFile("go.wav"))
        self.snd_jump.setSource(QUrl.fromLocalFile("jump.wav"))
        self.snd_finish.setSource(QUrl.fromLocalFile("finish.wav"))

        self.snd_go.setVolume(0.5)
        self.snd_jump.setVolume(0.5)
        self.snd_finish.setVolume(0.7)

        self.data_store = {}
        self.history_y = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

        self.left_score = 0
        self.right_score = 0
        self.winner = None

        self.current_frame = None

        self.icon1 = QPixmap("nl.png")
        self.icon2 = QPixmap("nl.png")
        self.icon_size = 140

        self.icon_happy = "happy.png"
        self.icon_sad = "sad.png"

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black;")

        self.race_label = QLabel()
        self.race_label.setFixedHeight(180)

        btn_play = QPushButton("Play")
        btn_pause = QPushButton("Pause")
        btn_reset = QPushButton("Reset")
        btn_capture_window = QPushButton("Open Capture Window")

        btn_play.clicked.connect(self.start)
        btn_pause.clicked.connect(self.stop)
        btn_reset.clicked.connect(self.reset)
        btn_capture_window.clicked.connect(self.open_capture_window)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_play)
        btn_layout.addWidget(btn_pause)
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(btn_capture_window)

        layout = QVBoxLayout()
        layout.addWidget(self.race_label)
        layout.addWidget(self.video_label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setWindowTitle("Jump Battle Race")

    def overlay_icon(self, frame, img_path, x, y, size=220):
        icon = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            return frame

        icon = cv2.resize(icon, (size, size))
        h, w = icon.shape[:2]

        x = int(x - w // 2)
        y = int(y - h // 2)

        if icon.shape[2] == 4:
            alpha = icon[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * icon[:, :, c] +
                    (1 - alpha) * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = icon

        return frame

    def open_capture_window(self):
        self.capture_window = CaptureWindow(self)
        self.capture_window.show()

    def start(self):
        if self.winner:
            return

        self.countdown = 3
        self.countdown_timer.start(1000)

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
        else:
            self.countdown_timer.stop()
            self.countdown = None

            now = time.time()
            self.left_start_time = now
            self.right_start_time = now

            self.snd_go.play()

            self.timer.start(50)

    def stop(self):
        self.timer.stop()

    def reset(self):
        self.left_score = 0
        self.right_score = 0
        self.winner = None
        self.data_store.clear()
        self.history_y.clear()

        self.left_start_time = None
        self.right_start_time = None
        self.left_time = 0
        self.right_time = 0

        self.update_race()

    def update_race(self):
        w = self.race_label.width()
        h = self.race_label.height()

        pix = QPixmap(w, h)
        pix.fill(Qt.black)

        painter = QPainter(pix)

        p1 = min(1.0, self.left_score / FINISH_SCORE)
        p2 = min(1.0, self.right_score / FINISH_SCORE)

        y1, y2 = 50, 120

        painter.setPen(QColor(100,100,100))
        painter.drawLine(0, y1, w, y1)
        painter.drawLine(0, y2, w, y2)

        painter.setPen(QColor("green"))
        painter.drawLine(0, y1, int(w*p1), y1)

        painter.setPen(QColor("blue"))
        painter.drawLine(0, y2, int(w*p2), y2)

        icon1 = self.icon1.scaled(self.icon_size, self.icon_size)
        icon2 = self.icon2.scaled(self.icon_size, self.icon_size)

        painter.drawPixmap(int(w * p1) - self.icon_size // 2, y1 - self.icon_size // 2, icon1)
        painter.drawPixmap(int(w * p2) - self.icon_size // 2, y2 - self.icon_size // 2, icon2)

        painter.setPen(QColor("white"))
        painter.drawLine(w-5, 0, w-5, h)

        painter.end()
        self.race_label.setPixmap(pix)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mid = w // 2

        # update waktu
        if self.left_start_time is not None and self.winner is None:
            self.left_time = time.time() - self.left_start_time
        if self.right_start_time is not None and self.winner is None:
            self.right_time = time.time() - self.right_start_time

        results = model.predict(frame, conf=0.5, device=device)

        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()

            players = []

            # ambil semua kandidat player (berdasarkan kaki)
            for kp in keypoints:
                la, ra = kp[15], kp[16]

                if (la == 0).all() and (ra == 0).all():
                    continue

                if (la == 0).all():
                    cx, cy = ra
                elif (ra == 0).all():
                    cx, cy = la
                else:
                    cx = (la[0] + ra[0]) / 2
                    cy = (la[1] + ra[1]) / 2

                players.append((cx, cy))

            # pilih hanya 1 kiri dan 1 kanan (paling dekat ke tengah)
            left_player = None
            right_player = None

            for (cx, cy) in players:
                if cx < mid:
                    if left_player is None or cx > left_player[0]:
                        left_player = (cx, cy)
                else:
                    if right_player is None or cx < right_player[0]:
                        right_player = (cx, cy)

            tracked = {
                "left": left_player,
                "right": right_player
            }

            for side, player in tracked.items():
                if player is None:
                    continue

                cx, cy = player

                self.history_y[side].append(cy)
                if len(self.history_y[side]) < HISTORY_LEN:
                    continue

                cy = sum(self.history_y[side]) / len(self.history_y[side])

                if side not in self.data_store:
                    self.data_store[side] = {
                        "y": cy,
                        "state": "ground",
                        "peak_y": cy
                    }
                    continue

                d = self.data_store[side]

                if d["state"] == "ground":
                    if cy < d["y"] - MIN_RISE:
                        d["state"] = "up"
                        d["start_y"] = cy
                        d["peak_y"] = cy

                elif d["state"] == "up":
                    if cy < d["peak_y"]:
                        d["peak_y"] = cy

                    if cy > d["peak_y"] + MIN_FALL:
                        jump_height = d["start_y"] - d["peak_y"]

                        if jump_height > MIN_JUMP:
                            self.snd_jump.play()

                            if side == "left":
                                self.left_score += 1
                            else:
                                self.right_score += 1

                            self.update_race()

                            if self.left_score >= FINISH_SCORE:
                                self.winner = "PLAYER 1 WIN"
                                self.snd_finish.play()
                                self.stop()

                            elif self.right_score >= FINISH_SCORE:
                                self.winner = "PLAYER 2 WIN"
                                self.snd_finish.play()
                                self.stop()

                        d["state"] = "ground"

                d["y"] = cy

                # titik merah tetap muncul
                cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)

        # countdown
        if self.countdown is not None:
            text = str(self.countdown + 1) if self.countdown > 0 else "GO!"
            cv2.putText(frame, text, (int(w*0.4), int(h*0.6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)

        # timer kiri kanan
        if self.left_start_time is not None:
            left_text = f"{self.left_time:.3f}s"
            right_text = f"{self.right_time:.3f}s"

            cv2.putText(frame, left_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            text_size = cv2.getTextSize(right_text,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

            cv2.putText(frame, right_text,
                        (w - text_size[0] - 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # =========================
            # 🔥 WINNER CHECK + DISPLAY (TARUH DI SINI)
            # =========================
            if not self.winner:
                if self.left_score >= FINISH_SCORE:
                    self.winner = "PLAYER 1 WIN"
                elif self.right_score >= FINISH_SCORE:
                    self.winner = "PLAYER 2 WIN"

            if self.winner:
                left_pos = (int(w * 0.25), int(h * 0.5))
                right_pos = (int(w * 0.75), int(h * 0.5))

                if self.winner == "PLAYER 1 WIN":
                    frame = self.overlay_icon(frame, self.icon_happy, *left_pos, 260)
                    frame = self.overlay_icon(frame, self.icon_sad, *right_pos, 260)

                    cv2.putText(frame, "WIN", left_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(frame, "LOSE", right_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4)
                else:
                    frame = self.overlay_icon(frame, self.icon_sad, *left_pos, 260)
                    frame = self.overlay_icon(frame, self.icon_happy, *right_pos, 260)

                    cv2.putText(frame, "LOSE", left_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4)
                    cv2.putText(frame, "WIN", right_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

        cv2.line(frame, (mid,0), (mid,h), (255,255,255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

app = QApplication(sys.argv)
window = JumpApp()
window.showMaximized()
sys.exit(app.exec())