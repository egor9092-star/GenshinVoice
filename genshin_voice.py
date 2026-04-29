import sys
import time
import re
import os
import json
import ctypes
import traceback
import numpy as np
import cv2
from PIL import ImageGrab
import easyocr
from difflib import SequenceMatcher
import threading
from queue import Queue
import torch
import sounddevice as sd
import requests

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QDesktopWidget,
    QComboBox, QProgressBar, QMessageBox, QTabWidget, QTextBrowser
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QImage

# ---------------------- НАСТРОЙКИ ----------------------
APP_NAME = "Genshin Voice"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
OFFICIAL_MODEL_URL = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'
SILERO_VOICES = ['aidar', 'baya', 'eugene', 'kseniya', 'xenia', 'random']
SAMPLE_RATE = 24000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------------------------------------------------------

voice_queue = Queue()
history = []
paused = False
tts_model = None
ocr_reader = None
sapi_voice = None
use_silero_voice = False
current_silero_speaker = SILERO_VOICES[0]

# --- вспомогательные функции ---
def improve_speech(text):
    if not text: return ""
    yo_map = {'еще': 'ещё', 'все': 'всё', 'идет': 'идёт', 'пришел': 'пришёл',
              'свое': 'своё', 'твое': 'твоё', 'ее': 'её', 'нее': 'неё'}
    for word, replacement in yo_map.items():
        text = re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)
    text = text.replace("...", " — ").replace(".", ". ")
    accents = {"Архонт": "Арх+онт", "Паймон": "П+аймон", "Ли Юэ": "Ли Ю+э"}
    for word, replacement in accents.items():
        text = text.replace(word, replacement)
    return text

def clean_ocr_text(text):
    text = re.sub(r'[^а-яА-ЯёЁ\s.,!?—\-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(mask)

# --- Перенаправление stdout ---
class StdoutRedirector(QObject):
    text_signal = pyqtSignal(str)
    def write(self, text):
        self.text_signal.emit(text)
    def flush(self):
        pass

# --- Окно выделения ---
class AreaSelector(QWidget):
    area_selected = pyqtSignal(QRect)
    cancelled = pyqtSignal()

    def __init__(self, background_pixmap):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        screen = QDesktopWidget().availableGeometry()
        self.setGeometry(screen)
        self.background = background_pixmap
        self.origin = QPoint()
        self.current_rect = QRect()
        self.showFullScreen()
        self._force_focus()

    def _force_focus(self):
        hwnd = int(self.winId())
        ctypes.windll.user32.ShowWindow(hwnd, 5)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        self.activateWindow()
        self.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))
        if not self.current_rect.isEmpty():
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(self.current_rect, self.background, self.current_rect)
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(self.current_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.current_rect = QRect(self.origin, self.origin)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.current_rect = QRect(self.origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.current_rect.isEmpty():
            self.area_selected.emit(self.current_rect)
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()

# --- Главное окно ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(750, 650)

        self.dialogue_area = None
        self.paused = False
        self.calibration_active = False
        self.calibration_countdown = 0
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self._calibration_tick)

        # Загрузка конфигурации (устанавливает use_silero_voice, current_silero_speaker, область)
        self.load_config()

        # Инициализация SAPI
        self.init_sapi()

        # Центральный виджет с вкладками
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Вкладка "Main" ---
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)

        gpu_text = "CUDA - OK" if DEVICE.type == 'cuda' else "CUDA NOT AVAILABLE (CPU)"
        self.gpu_label = QLabel(f"GPU: {gpu_text}")
        self.gpu_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.gpu_label)

        self.status_label = QLabel("Статус: Готов (Windows Voice)")
        layout.addWidget(self.status_label)

        # Управление голосом
        voice_sel_layout = QHBoxLayout()
        self.win_btn = QPushButton("🔊 Windows Voice")
        self.win_btn.setCheckable(True)
        self.win_btn.setChecked(not use_silero_voice)
        self.win_btn.clicked.connect(self.on_win_clicked)
        voice_sel_layout.addWidget(self.win_btn)

        self.silero_btn = QPushButton("🤖 Silero Voice")
        self.silero_btn.setCheckable(True)
        self.silero_btn.setChecked(use_silero_voice)
        self.silero_btn.setEnabled(True)
        self.silero_btn.clicked.connect(self.on_silero_clicked)
        voice_sel_layout.addWidget(self.silero_btn)
        layout.addLayout(voice_sel_layout)

        # Выбор голоса Silero
        silero_sub_layout = QHBoxLayout()
        silero_sub_layout.addWidget(QLabel("Голос Silero:"))
        self.silero_combo = QComboBox()
        self.silero_combo.addItems(SILERO_VOICES)
        idx = self.silero_combo.findText(current_silero_speaker)
        if idx >= 0:
            self.silero_combo.setCurrentIndex(idx)
        self.silero_combo.currentIndexChanged.connect(self.on_silero_speaker_changed)
        self.silero_combo.setEnabled(use_silero_voice)
        silero_sub_layout.addWidget(self.silero_combo)
        layout.addLayout(silero_sub_layout)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("📷 Калибровать область")
        self.calibrate_btn.clicked.connect(self.start_calibration_countdown)
        btn_layout.addWidget(self.calibrate_btn)

        self.pause_btn = QPushButton("⏯️ Пауза")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)
        layout.addLayout(btn_layout)

        # Лог
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_widget)

        # Строка состояния
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status_bar()

        # --- Вкладка "About" ---
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_text = QTextBrowser()
        about_text.setOpenExternalLinks(True)
        about_text.setHtml(self._get_about_html())
        about_layout.addWidget(about_text)

        self.tabs.addTab(main_tab, "Main")
        self.tabs.addTab(about_tab, "About")

        # Перенаправление stdout
        self.stdout_redirector = StdoutRedirector()
        self.stdout_redirector.text_signal.connect(self.log_widget.append)
        sys.stdout = self.stdout_redirector

        # Перехват исключений
        def gui_excepthook(exc_type, exc_value, tb):
            msg = ''.join(traceback.format_exception(exc_type, exc_value, tb))
            self.log_widget.append(f"КРИТИЧЕСКАЯ ОШИБКА:\n{msg}")
            sys.__excepthook__(exc_type, exc_value, tb)
        sys.excepthook = gui_excepthook

        # Загрузка модели Silero, если она была выбрана ранее
        self.init_tts_model_if_needed()

        # OCR и запуск потоков
        self.init_ocr()
        self.start_worker_threads()
        print("Программа готова.")

    # --- Работа с конфигурацией ---
    def load_config(self):
        global use_silero_voice, current_silero_speaker
        default_config = {
            "dialogue_area": None,
            "use_silero_voice": False,
            "current_silero_speaker": SILERO_VOICES[0]
        }
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                self.dialogue_area = config['dialogue_area']
                use_silero_voice = config['use_silero_voice']
                current_silero_speaker = config.get('current_silero_speaker', SILERO_VOICES[0])
                if current_silero_speaker not in SILERO_VOICES:
                    current_silero_speaker = SILERO_VOICES[0]
                if use_silero_voice and not os.path.exists(MODEL_PATH):
                    use_silero_voice = False
                    print("Модель Silero не найдена, переключение на Windows Voice.")
            else:
                self.dialogue_area = None
                use_silero_voice = False
                current_silero_speaker = SILERO_VOICES[0]
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            self.dialogue_area = None
            use_silero_voice = False
            current_silero_speaker = SILERO_VOICES[0]

    def save_config(self):
        config = {
            "dialogue_area": self.dialogue_area,
            "use_silero_voice": use_silero_voice,
            "current_silero_speaker": current_silero_speaker
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def init_tts_model_if_needed(self):
        """Если в настройках выбран Silero и модель существует, загружаем её сразу."""
        global tts_model, use_silero_voice
        if use_silero_voice and os.path.exists(MODEL_PATH) and tts_model is None:
            try:
                print("Загружаю модель Silero в память (из конфига)...")
                tts_model = torch.package.PackageImporter(MODEL_PATH).load_pickle('tts_models', 'model')
                tts_model.to(DEVICE)
                print("Модель Silero готова.")
            except Exception as e:
                print(f"Ошибка загрузки модели Silero при старте: {e}")
                use_silero_voice = False  # сбрасываем на Windows
                self.save_config()
        # Обновляем интерфейс в соответствии с текущим состоянием
        if use_silero_voice:
            self.win_btn.setChecked(False)
            self.silero_btn.setChecked(True)
            self.silero_combo.setEnabled(True)
            self.status_label.setText(f"Статус: Silero Voice ({current_silero_speaker})")
        else:
            self.win_btn.setChecked(True)
            self.silero_btn.setChecked(False)
            self.silero_combo.setEnabled(False)
            self.status_label.setText("Статус: Windows Voice")

    def _get_about_html(self):
        return """
        <h2>Genshin Voice</h2>
        <p><b>Dialogue voiceover tool for Genshin Impact</b></p>
        <p>
        This program is free software distributed under the terms of the
        <a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU General Public License v3.0 (GPLv3)</a>.
        You may freely use, modify, and distribute this program, provided that any
        derivative works are also distributed under the same license.
        </p>
        <h3>Third-party components and licenses</h3>
        <ul>
        <li><b>Silero TTS</b> (model <code>v4_ru.pt</code>) — 
            <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.
            The neural network voice model is NOT bundled with this application.
            Users must obtain the model independently. This tool is intended for
            <b>strictly non-commercial, personal use</b> in compliance with Silero’s terms.
            <b>This project does not generate revenue.</b>
        </li>
        <li><b>EasyOCR</b> — 
            <a href="https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE">Apache License 2.0</a>
        </li>
        <li><b>PyQt5</b> — 
            <a href="https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html">GPL v3</a>
        </li>
        <li><b>PyTorch</b> — 
            <a href="https://github.com/pytorch/pytorch/blob/main/LICENSE">BSD-3-Clause</a>
        </li>
        <li><b>sounddevice</b> — 
            <a href="https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE">MIT</a>
        </li>
        <li><b>OpenCV</b> — 
            <a href="https://opencv.org/license/">Apache License 2.0</a>
        </li>
        <li><b>Pillow (PIL)</b> — 
            <a href="https://raw.githubusercontent.com/python-pillow/Pillow/main/LICENSE">HPND</a>
        </li>
        <li><b>NumPy</b> — 
            <a href="https://numpy.org/doc/stable/license.html">BSD-3-Clause</a>
        </li>
        <li><b>Requests</b> — 
            <a href="https://github.com/psf/requests/blob/main/LICENSE">Apache License 2.0</a>
        </li>
        <li><b>pywin32</b> — 
            <a href="https://github.com/mhammond/pywin32/blob/main/PyWin32/License.txt">PSF</a>
        </li>
        </ul>
        <h3>Disclaimer regarding Genshin Impact</h3>
        <p>
        This software is not affiliated with, endorsed by, or supported by HoYoverse.
        </p>
        <p>
        <b>Method:</b> This tool uses <b>OCR (Screen Capturing)</b> technology to read text
        from the screen. It <b>does not inject code, modify game files, or read the game's
        internal memory</b>. No copyrighted assets (audio, images, or code) from Genshin Impact
        are included in this package.
        </p>
        <p>
        The extracted text is used exclusively for immediate voice synthesis and is never
        transmitted, stored, or processed for any other purpose.
        </p>
        <p>
        <b>Use at your own risk.</b> The author is not responsible for any actions taken by
        HoYoverse regarding user accounts.
        </p>
        <h3>Source code</h3>
        <p>
        The source code of Genshin Voice is available at:
        <a href="https://github.com/egor9092-star/genshin-voice">https://github.com/egor9092-star/genshin-voice</a>.
        </p>
        """

    def init_sapi(self):
        global sapi_voice
        try:
            import win32com.client
            sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            for voice in sapi_voice.GetVoices():
                if 'Russian' in voice.GetDescription() or 'русский' in voice.GetDescription().lower():
                    sapi_voice.Voice = voice
                    break
            print("Голос Windows (SAPI) инициализирован.")
        except Exception as e:
            print(f"Ошибка инициализации SAPI: {e}")
            sapi_voice = None

    def init_ocr(self):
        global ocr_reader
        try:
            print("Загрузка EasyOCR...")
            ocr_reader = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())
            print("EasyOCR загружен.")
        except Exception as e:
            print(f"Ошибка загрузки EasyOCR: {e}")

    def start_worker_threads(self):
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=recognition_worker, args=(self,), daemon=True).start()

    # --- Управление голосом ---
    def on_win_clicked(self, checked):
        global use_silero_voice
        if checked:
            self.win_btn.setChecked(True)
            self.silero_btn.setChecked(False)
            use_silero_voice = False
            self.silero_combo.setEnabled(False)
            self.status_label.setText("Статус: Windows Voice")
            print("Переключено на Windows Voice")
            self.save_config()
        else:
            if not self.silero_btn.isChecked():
                self.win_btn.setChecked(True)
                print("Нельзя отключить оба голоса. Windows Voice остаётся активным.")

    def on_silero_clicked(self, checked):
        global use_silero_voice, tts_model
        if not checked:
            if not self.win_btn.isChecked():
                self.silero_btn.setChecked(True)
                print("Нельзя отключить оба голоса.")
            return

        if not os.path.exists(MODEL_PATH):
            self.ask_download_model()
            self.silero_btn.setChecked(False)
            return

        if tts_model is None:
            try:
                print("Загружаю модель Silero в память...")
                tts_model = torch.package.PackageImporter(MODEL_PATH).load_pickle('tts_models', 'model')
                tts_model.to(DEVICE)
                print("Модель Silero готова.")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                self.silero_btn.setChecked(False)
                return

        self.win_btn.setChecked(False)
        use_silero_voice = True
        self.silero_combo.setEnabled(True)
        self.status_label.setText(f"Статус: Silero Voice ({current_silero_speaker})")
        print(f"Переключено на Silero, голос {current_silero_speaker}")
        self.save_config()

    def on_silero_speaker_changed(self, idx):
        global current_silero_speaker
        current_silero_speaker = self.silero_combo.itemText(idx)
        if use_silero_voice:
            self.status_label.setText(f"Статус: Silero Voice ({current_silero_speaker})")
            print(f"Голос Silero изменён на {current_silero_speaker}")
        self.save_config()

    def ask_download_model(self):
        reply = QMessageBox.question(self, "Модель Silero не найдена",
            "Файл модели не обнаружен. Загрузить его сейчас с официального сайта?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.download_silero_model()
        else:
            print("Загрузка отменена пользователем.")

    def download_silero_model(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        print(f"Скачивание модели с {OFFICIAL_MODEL_URL}...")
        print(f"Файл будет сохранён в {MODEL_PATH}")

        def do_download():
            global tts_model, use_silero_voice
            try:
                response = requests.get(OFFICIAL_MODEL_URL, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024
                downloaded = 0
                with open(MODEL_PATH, 'wb') as f:
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        f.write(data)
                        percent = int(100 * downloaded / total_size)
                        self.progress_bar.setValue(percent)
                        QApplication.processEvents()
                print("Загрузка завершена. Загружаем модель в память...")
                self.progress_bar.setValue(100)

                tts_model = torch.package.PackageImporter(MODEL_PATH).load_pickle('tts_models', 'model')
                tts_model.to(DEVICE)
                print("Модель Silero загружена и готова к работе.")

                def activate_silero():
                    self.silero_btn.setChecked(True)
                    self.win_btn.setChecked(False)
                    use_silero_voice = True
                    self.silero_combo.setEnabled(True)
                    self.status_label.setText(f"Статус: Silero Voice ({current_silero_speaker})")
                    print(f"Автоматически выбран Silero с голосом {current_silero_speaker}")
                    self.save_config()
                QTimer.singleShot(0, activate_silero)

            except Exception as e:
                print(f"Ошибка при загрузке модели: {e}")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
            finally:
                self.progress_bar.setVisible(False)

        threading.Thread(target=do_download, daemon=True).start()

    # --- Калибровка ---
    def start_calibration_countdown(self):
        if self.calibration_active:
            return
        self.calibration_active = True
        self.calibration_countdown = 3
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setText(f"Калибровка: {self.calibration_countdown} сек...")
        self.hide()
        self.calibration_timer.start(1000)

    def _calibration_tick(self):
        self.calibration_countdown -= 1
        if self.calibration_countdown > 0:
            self.calibrate_btn.setText(f"Калибровка: {self.calibration_countdown} сек...")
        else:
            self.calibration_timer.stop()
            self._do_calibrate()

    def _do_calibrate(self):
        print("Делаю скриншот экрана...")
        try:
            screenshot = ImageGrab.grab()
            img = screenshot.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"Ошибка скриншота: {e}")
            self._finish_calibration()
            return

        self.selector = AreaSelector(pixmap)
        self.selector.area_selected.connect(self.on_area_selected)
        self.selector.cancelled.connect(self.on_calibration_cancelled)
        self.selector.show()

    def on_area_selected(self, rect: QRect):
        self.dialogue_area = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
        self.update_status_bar()
        print(f"Новая область: {self.dialogue_area}")
        self._finish_calibration()
        self.save_config()

    def on_calibration_cancelled(self):
        print("Калибровка отменена.")
        self._finish_calibration()

    def _finish_calibration(self):
        self.calibration_active = False
        self.calibrate_btn.setText("📷 Калибровать область")
        self.calibrate_btn.setEnabled(True)
        self.show()

    def toggle_pause(self, checked):
        global paused
        paused = checked
        if checked:
            self.pause_btn.setText("▶️ Продолжить")
            self.status_label.setText("Статус: Пауза")
        else:
            self.pause_btn.setText("⏯️ Пауза")
            status = "Silero Voice" if use_silero_voice else "Windows Voice"
            self.status_label.setText(f"Статус: Активен ({status})")

    def update_status_bar(self):
        if self.dialogue_area:
            self.status_bar.showMessage(f"Область: {self.dialogue_area}")
        else:
            self.status_bar.showMessage("Область не задана. Нажмите «Калибровать область».")

# --- Поток TTS ---
def tts_worker():
    global paused, tts_model, sapi_voice, use_silero_voice, current_silero_speaker
    while True:
        text = voice_queue.get()
        if text is None:
            break
        while paused:
            time.sleep(0.1)
        try:
            prepared_text = improve_speech(text)
            if not prepared_text.endswith(('.', '!', '?', '—')):
                prepared_text += '.'

            if use_silero_voice and tts_model is not None:
                start_tts = time.perf_counter()
                with torch.no_grad():
                    audio = tts_model.apply_tts(
                        text=prepared_text,
                        speaker=current_silero_speaker,
                        sample_rate=SAMPLE_RATE
                    )
                end_tts = time.perf_counter()
                print(f"Синтез (Silero, {current_silero_speaker}): {(end_tts - start_tts)*1000:.0f} мс")
                sd.play(audio.numpy(), SAMPLE_RATE)
                sd.wait()
            else:
                if sapi_voice is not None:
                    start_tts = time.perf_counter()
                    sapi_voice.Speak(prepared_text, 1)
                    while sapi_voice.Status.RunningState != 1:
                        time.sleep(0.05)
                    end_tts = time.perf_counter()
                    print(f"Синтез (SAPI): {(end_tts - start_tts)*1000:.0f} мс")
                else:
                    print("Нет доступного голосового движка!")
        except Exception as e:
            print(f"Ошибка TTS: {e}")
        voice_queue.task_done()

# --- Поток распознавания ---
def recognition_worker(main_window):
    global paused, history, ocr_reader
    last_raw_text = ""
    stable_counter = 0
    print("Поток распознавания запущен.")
    while True:
        if paused:
            time.sleep(0.1)
            continue
        try:
            from ctypes import windll, create_unicode_buffer
            hwnd = windll.user32.GetForegroundWindow()
            buf = create_unicode_buffer(512)
            windll.user32.GetWindowTextW(hwnd, buf, 512)
            if "Genshin Impact" not in buf.value:
                time.sleep(0.1)
                continue

            if not main_window.dialogue_area:
                time.sleep(0.1)
                continue

            img = ImageGrab.grab(bbox=main_window.dialogue_area)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_img = preprocess_image(img_bgr)

            start_ocr = time.perf_counter()
            result = ocr_reader.readtext(processed_img, detail=0, paragraph=True)
            end_ocr = time.perf_counter()

            if result:
                current_text = " ".join(result).strip()
                if current_text == last_raw_text and len(current_text) > 3:
                    stable_counter += 1
                else:
                    stable_counter, last_raw_text = 0, current_text

                if stable_counter == 2:
                    print(f"OCR: {(end_ocr - start_ocr)*1000:.0f} мс")
                    clean = clean_ocr_text(current_text)
                    if len(clean) > 3 and not (history and SequenceMatcher(None, clean, history[-1]).ratio() > 0.8):
                        print(f">>> {clean}")
                        voice_queue.put(clean)
                        history.append(clean)
                        if len(history) > 10:
                            history.pop(0)
                        stable_counter = 0
            else:
                stable_counter, last_raw_text = 0, ""
        except Exception as e:
            print(f"Ошибка распознавания: {e}")
        time.sleep(0.1)

# --- Запуск ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())