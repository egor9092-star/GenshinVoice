import sys, os
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import time
import re
import json
import ctypes
import traceback
import numpy as np
import cv2
from PIL import ImageGrab
import pytesseract
from difflib import SequenceMatcher
import threading
from queue import Queue
import sounddevice as sd
import requests

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QDesktopWidget,
    QComboBox, QProgressBar, QMessageBox, QTabWidget, QTextBrowser, QFileDialog
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QObject, QTimer, Q_ARG
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QImage

# ---------------------- НАСТРОЙКИ ----------------------
APP_NAME = "Genshin Voice"

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pt')
OFFICIAL_MODEL_URL = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'
SILERO_VOICES = ['aidar', 'baya', 'eugene', 'kseniya', 'xenia', 'random']
SAMPLE_RATE = 48000
DEVICE = 'cpu'

# ----------- НАСТРОЙКА TESSERACT ------------------------
def find_tesseract():
    local = os.path.join(BASE_DIR, 'bin', 'tesseract', 'tesseract.exe')
    if os.path.isfile(local):
        return local
    internal = os.path.join(BASE_DIR, '_internal', 'bin', 'tesseract', 'tesseract.exe')
    if os.path.isfile(internal):
        return internal
    return None

def configure_tesseract(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    tessdata_dir = os.path.join(os.path.dirname(tesseract_path), 'tessdata')
    os.environ['TESSDATA_PREFIX'] = tessdata_dir
    return tessdata_dir

TESSERACT_PATH = find_tesseract()
if TESSERACT_PATH:
    configure_tesseract(TESSERACT_PATH)
# -------------------------------------------------------

voice_queue = Queue()
history = []
last_voiced_text = ""
paused = False
tts_model = None
sapi_voice = None
use_silero_voice = False
current_silero_speaker = SILERO_VOICES[0]
genshin_active = False

# --- вспомогательные функции ---
def improve_speech(text):
    if not text:
        return ""
    brackets = ['<<', '>>', '«', '»', '"', '"', '\u201c', '\u201d', '(', ')', '[', ']', '{', '}', '<', '>']
    for b in brackets:
        text = text.replace(b, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    number_words = {'0':'ноль','1':'один','2':'два','3':'три','4':'четыре','5':'пять','6':'шесть','7':'семь','8':'восемь','9':'девять','10':'десять'}
    for digit, word in number_words.items():
        text = re.sub(rf'(?<!\d){digit}(?!\d)', word, text)
    ocr_fixes = {'Дзынь-лащ':'Дзынь-клац','Дзынь лащ':'Дзынь-клац'}
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)
    yo_map = {'еще':'ещё','идет':'идёт','пришел':'пришёл','свое':'своё','твое':'твоё','ее':'её','нее':'неё'}
    for word, replacement in yo_map.items():
        text = re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)
    text = text.replace("...", " — ").replace(".", ". ")
    accents = {"Архонт":"Арх+онт","Паймон":"П+аймон","Ли Юэ":"Ли Ю+э","большая":"больш+ая","Большая":"Больш+ая","вести":"вест+и","Вести":"Вест+и","стариком":"старик+ом","Стариком":"Старик+ом","Ему":"Е+му","ему":"е+му","никого":"никог+о","Никого":"Никог+о"}
    for word, replacement in accents.items():
        text = text.replace(word, replacement)
    return text

def clean_ocr_text(text):
    text = re.sub(r'[^а-яА-ЯёЁ0-9R\s.,!?—\-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_text_from_image(img_bgr):
    if img_bgr is None:
        return ""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    clean = cv2.bitwise_not(mask)
    upscaled = cv2.resize(clean, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r'--oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(binary, lang='rus', config=config)
        return text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"Ошибка Tesseract: {e}")
        return ""

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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
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
        painter.fillRect(self.rect(), QColor(0,0,0,160))
        if not self.current_rect.isEmpty():
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(self.current_rect, self.background, self.current_rect)
            painter.setPen(QPen(QColor(255,255,0), 2))
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
        self.resize(750, 700)
        self.dialogue_area = None
        self.paused = False
        self.calibration_active = False
        self.calibration_countdown = 0
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self._calibration_tick)
        self.load_config()
        self.init_sapi()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        gpu_layout = QHBoxLayout()
        gpu_text = "CUDA NOT AVAILABLE (CPU)"
        self.gpu_label = QLabel(f"GPU: {gpu_text}")
        self.gpu_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.gpu_label)
        gpu_layout.addStretch()
        self.game_status_label = QLabel("Genshin Impact: проверка...")
        self.game_status_label.setFont(QFont("Arial", 10))
        gpu_layout.addWidget(self.game_status_label)
        layout.addLayout(gpu_layout)
        self.status_label = QLabel("Статус: Готов (Windows Voice)")
        layout.addWidget(self.status_label)
        tess_layout = QHBoxLayout()
        tess_layout.addWidget(QLabel("Tesseract:"))
        self.tess_path_label = QLabel("Не найден")
        self.tess_path_label.setStyleSheet("color: red;")
        tess_layout.addWidget(self.tess_path_label)
        self.tess_btn = QPushButton("📂 Указать Tesseract")
        self.tess_btn.setToolTip("Укажите полный путь к tesseract.exe.\nПапка tessdata должна находиться рядом с этим файлом.")
        self.tess_btn.clicked.connect(self.select_tesseract)
        tess_layout.addWidget(self.tess_btn)
        layout.addLayout(tess_layout)
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        btn_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("📷 Калибровать область")
        self.calibrate_btn.clicked.connect(self.start_calibration_countdown)
        btn_layout.addWidget(self.calibrate_btn)
        self.pause_btn = QPushButton("⏯️ Пауза")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)
        layout.addLayout(btn_layout)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_widget)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status_bar()
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_text = QTextBrowser()
        about_text.setOpenExternalLinks(True)
        about_text.setHtml(self._get_about_html())
        about_layout.addWidget(about_text)
        self.tabs.addTab(main_tab, "Main")
        self.tabs.addTab(about_tab, "About")
        self.stdout_redirector = StdoutRedirector()
        self.stdout_redirector.text_signal.connect(self.log_widget.append)
        sys.stdout = self.stdout_redirector
        def gui_excepthook(exc_type, exc_value, tb):
            msg = ''.join(traceback.format_exception(exc_type, exc_value, tb))
            self.log_widget.append(f"КРИТИЧЕСКАЯ ОШИБКА:\n{msg}")
            sys.__excepthook__(exc_type, exc_value, tb)
        sys.excepthook = gui_excepthook
        self.init_tts_model_if_needed()
        self.update_tesseract_status()
        self.update_genshin_status(False)
        self.game_check_timer = QTimer()
        self.game_check_timer.timeout.connect(self.check_game_window)
        self.game_check_timer.start(500)
        self.start_worker_threads()
        self.check_game_window()
        QTimer.singleShot(0, self.show_tesseract_help_if_needed)
        print("Программа готова (Tesseract).")

    # ===== ВАЖНО: метод, добавляющий путь к библиотекам torch =====
    def _ensure_torch_path(self):
        """Добавляет папку torch\lib в PATH, чтобы c10.dll и другие DLL могли загрузиться."""
        try:
            import torch
            torch_path = os.path.dirname(torch.__file__)
            torch_lib = os.path.join(torch_path, 'lib')
        except ImportError:
            # Если torch ещё не импортирован, ищем в sys.path
            for p in sys.path:
                candidate = os.path.join(p, 'torch', 'lib')
                if os.path.isdir(candidate):
                    torch_lib = candidate
                    break
            else:
                # Запасной вариант для замороженного приложения
                base = sys._MEIPASS if getattr(sys, 'frozen', False) else BASE_DIR
                torch_lib = os.path.join(base, 'torch', 'lib')
        if os.path.isdir(torch_lib):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

    # ===== Загрузка конфига и прочее =====
    def load_config(self):
        global use_silero_voice, current_silero_speaker, TESSERACT_PATH
        default_config = {"dialogue_area": None, "use_silero_voice": False, "current_silero_speaker": SILERO_VOICES[0], "tesseract_path": ""}
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = default_config
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            self.dialogue_area = config['dialogue_area']
            use_silero_voice = config['use_silero_voice']
            current_silero_speaker = config.get('current_silero_speaker', SILERO_VOICES[0])
            if current_silero_speaker not in SILERO_VOICES:
                current_silero_speaker = SILERO_VOICES[0]
            saved_tess_path = config.get('tesseract_path', '')
            if saved_tess_path and os.path.isfile(saved_tess_path):
                TESSERACT_PATH = saved_tess_path
                configure_tesseract(TESSERACT_PATH)
            if use_silero_voice and not os.path.exists(MODEL_PATH):
                use_silero_voice = False
                print("Модель Silero не найдена, переключение на Windows Voice.")
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            self.dialogue_area = None
            use_silero_voice = False
            current_silero_speaker = SILERO_VOICES[0]

    def save_config(self):
        config = {"dialogue_area": self.dialogue_area, "use_silero_voice": use_silero_voice, "current_silero_speaker": current_silero_speaker, "tesseract_path": TESSERACT_PATH if TESSERACT_PATH else ""}
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def init_tts_model_if_needed(self):
        global tts_model, use_silero_voice
        if use_silero_voice and os.path.exists(MODEL_PATH) and tts_model is None:
            try:
                self._ensure_torch_path()   # <-- ВЫЗОВ ПЕРЕД ИМПОРТОМ
                import torch
                print("Загружаю модель Silero в память...")
                tts_model = torch.package.PackageImporter(MODEL_PATH).load_pickle('tts_models', 'model')
                tts_model.to(DEVICE)
                print("Модель Silero готова.")
            except Exception as e:
                print(f"Ошибка загрузки модели Silero при старте: {e}")
                use_silero_voice = False
                self.save_config()
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

    def on_silero_clicked(self, checked):
        global use_silero_voice, tts_model
        if not checked:
            if not self.win_btn.isChecked():
                self.silero_btn.setChecked(True)
            return
        if not os.path.exists(MODEL_PATH):
            self.ask_download_model()
            self.silero_btn.setChecked(False)
            return
        if tts_model is None:
            try:
                self._ensure_torch_path()   # <-- ВЫЗОВ
                import torch
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

    def download_silero_model(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        print(f"Скачивание модели с {OFFICIAL_MODEL_URL}...")
        def do_download():
            global tts_model, use_silero_voice
            try:
                response = requests.get(OFFICIAL_MODEL_URL, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024*1024
                downloaded = 0
                with open(MODEL_PATH, 'wb') as f:
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        f.write(data)
                        percent = int(100 * downloaded / total_size)
                        self.progress_bar.setValue(percent)
                        QApplication.processEvents()
                self.progress_bar.setValue(100)
                self._ensure_torch_path()   # <-- ВЫЗОВ
                import torch
                tts_model = torch.package.PackageImporter(MODEL_PATH).load_pickle('tts_models', 'model')
                tts_model.to(DEVICE)
                print("Модель Silero загружена.")
                QTimer.singleShot(0, self.activate_silero_after_download)
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
            finally:
                self.progress_bar.setVisible(False)
        threading.Thread(target=do_download, daemon=True).start()

    def activate_silero_after_download(self):
        self.silero_btn.setChecked(True)
        self.win_btn.setChecked(False)
        use_silero_voice = True
        self.silero_combo.setEnabled(True)
        self.status_label.setText(f"Статус: Silero Voice ({current_silero_speaker})")
        self.save_config()

    # --- калибровка ---
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
            data = img.tobytes("raw","RGB")
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
        self.dialogue_area = (rect.x(), rect.y(), rect.x()+rect.width(), rect.y()+rect.height())
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
            self.status_bar.showMessage("Область – ОК")
            self.status_bar.setStyleSheet("color: green;")
        else:
            self.status_bar.showMessage("Область не захвачена, откалибруйте область")
            self.status_bar.setStyleSheet("color: red;")

    def show_tesseract_help_if_needed(self):
        if TESSERACT_PATH and os.path.isfile(TESSERACT_PATH):
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Tesseract не найден")
        msg.setTextFormat(Qt.RichText)
        msg.setText("<h3>Tesseract OCR не обнаружен</h3><p>Укажите путь к файлу <b>tesseract.exe</b>.</p><p>Обычно он находится в папке <code>bin/tesseract/</code>.</p><p>Рядом с <code>tesseract.exe</code> должна быть папка <b>tessdata</b> с языковыми файлами (например, <code>rus.traineddata</code>).</p><p>Нажмите кнопку <b>«📂 Указать Tesseract»</b>, чтобы выбрать файл вручную.</p>")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def update_genshin_status(self, active):
        global genshin_active
        genshin_active = active
        if active:
            self.game_status_label.setText("Genshin Impact активен")
            self.game_status_label.setStyleSheet("color: green;")
        else:
            self.game_status_label.setText("GenshinImpact.exe не активен")
            self.game_status_label.setStyleSheet("color: red;")

    def check_game_window(self):
        active = False
        try:
            from ctypes import windll, create_unicode_buffer
            hwnd = windll.user32.GetForegroundWindow()
            buf = create_unicode_buffer(512)
            windll.user32.GetWindowTextW(hwnd, buf, 512)
            if "Genshin Impact" in buf.value:
                active = True
        except:
            pass
        self.update_genshin_status(active)

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

    def start_worker_threads(self):
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=recognition_worker, args=(self,), daemon=True).start()

    def select_tesseract(self):
        path, _ = QFileDialog.getOpenFileName(self, "Укажите tesseract.exe", "", "Executable (*.exe)")
        if path and os.path.isfile(path):
            self.set_tesseract_path(path)

    def set_tesseract_path(self, path):
        global TESSERACT_PATH
        TESSERACT_PATH = path
        configure_tesseract(path)
        self.update_tesseract_status()
        self.save_config()
        print(f"Tesseract установлен: {path}")

    def update_tesseract_status(self):
        if TESSERACT_PATH and os.path.isfile(TESSERACT_PATH):
            tessdata = os.environ.get('TESSDATA_PREFIX', 'неизвестно')
            self.tess_path_label.setText(f"OK: {tessdata}")
            self.tess_path_label.setStyleSheet("color: green;")
        else:
            self.tess_path_label.setText("Не найден")
            self.tess_path_label.setStyleSheet("color: red;")

    def _get_about_html(self):
        html = ('<h2>Genshin Voice</h2>\n<p><b>Dialogue voiceover tool for Genshin Impact</b></p>\n<p>This program is free software distributed under the terms of the\n<a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU General Public License v3.0 (GPLv3)</a>.\nYou may freely use, modify, and distribute this program, provided that any\nderivative works are also distributed under the same license.</p>\n<h3>Third-party components and licenses</h3>\n<ul>\n<li><b>Silero TTS</b> (model <code>v4_ru.pt</code>) — <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>. The neural network voice model is NOT bundled with this application. Users must obtain the model independently. This tool is intended for <b>strictly non-commercial, personal use</b> in compliance with Silero’s terms. <b>This project does not generate revenue.</b></li>\n<li><b>Tesseract OCR</b> — <a href="https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE">Apache License 2.0</a></li>\n<li><b>PyQt5</b> — <a href="https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html">GPL v3</a></li>\n<li><b>PyTorch</b> — <a href="https://github.com/pytorch/pytorch/blob/main/LICENSE">BSD-3-Clause</a></li>\n<li><b>sounddevice</b> — <a href="https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE">MIT</a></li>\n<li><b>OpenCV</b> — <a href="https://opencv.org/license/">Apache License 2.0</a></li>\n<li><b>Pillow (PIL)</b> — <a href="https://raw.githubusercontent.com/python-pillow/Pillow/main/LICENSE">HPND</a></li>\n<li><b>NumPy</b> — <a href="https://numpy.org/doc/stable/license.html">BSD-3-Clause</a></li>\n<li><b>Requests</b> — <a href="https://github.com/psf/requests/blob/main/LICENSE">Apache License 2.0</a></li>\n<li><b>pywin32</b> — <a href="https://github.com/mhammond/pywin32/blob/main/PyWin32/License.txt">PSF</a></li>\n</ul>\n<h3>Disclaimer regarding Genshin Impact</h3>\n<p>This software is not affiliated with, endorsed by, or supported by HoYoverse.</p>\n<p><b>Method:</b> This tool uses <b>OCR (Screen Capturing)</b> technology to read text from the screen. It <b>does not inject code, modify game files, or read the game\'s internal memory</b>. No copyrighted assets (audio, images, or code) from Genshin Impact are included in this package.</p>\n<p>The extracted text is used exclusively for immediate voice synthesis and is never transmitted, stored, or processed for any other purpose.</p>\n<p><b>Use at your own risk.</b> The author is not responsible for any actions taken by HoYoverse regarding user accounts.</p>\n<h3>Source code</h3>\n<p>The source code of Genshin Voice is available at:\n<a href="https://github.com/egor9092-star/GenshinVoice">https://github.com/egor9092-star/GenshinVoice</a>.</p>')
        return html

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
                # Добавляем путь к DLL перед импортом torch
                if getattr(sys, 'frozen', False):
                    base = sys._MEIPASS
                else:
                    base = BASE_DIR
                torch_lib = os.path.join(base, 'torch', 'lib')
                if os.path.isdir(torch_lib):
                    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
                import torch
                start_tts = time.perf_counter()
                with torch.no_grad():
                    audio = tts_model.apply_tts(
                        text=prepared_text,
                        speaker=current_silero_speaker,
                        sample_rate=SAMPLE_RATE
                    )
                print(f"Синтез (Silero, {current_silero_speaker}): {(time.perf_counter() - start_tts)*1000:.0f} мс")
                sd.play(audio.numpy(), SAMPLE_RATE)
                sd.wait()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                if sapi_voice is not None:
                    start_tts = time.perf_counter()
                    sapi_voice.Speak(prepared_text, 1)
                    while sapi_voice.Status.RunningState != 1:
                        time.sleep(0.05)
                    print(f"Синтез (SAPI): {(time.perf_counter() - start_tts)*1000:.0f} мс")
                else:
                    print("Нет доступного голосового движка!")
        except Exception as e:
            print(f"Ошибка TTS: {e}")
        voice_queue.task_done()

# --- Поток распознавания ---
def recognition_worker(main_window):
    global paused, history, last_voiced_text, genshin_active
    last_raw_text = ""
    stable_counter = 0
    print("Поток распознавания (Tesseract) запущен.")
    while True:
        if paused:
            time.sleep(0.1)
            continue
        try:
            if not genshin_active:
                stable_counter, last_raw_text = 0, ""
                time.sleep(0.2)
                continue
            if not main_window.dialogue_area:
                time.sleep(0.1)
                continue
            img = ImageGrab.grab(bbox=main_window.dialogue_area)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            current_text = get_text_from_image(img_bgr)
            if last_voiced_text and SequenceMatcher(None, current_text.strip(), last_voiced_text).ratio() > 0.85:
                current_text = last_raw_text if last_raw_text else current_text
            if current_text:
                current_text = current_text.strip()
                if current_text == last_raw_text and len(current_text) > 3:
                    stable_counter += 1
                else:
                    stable_counter, last_raw_text = 0, current_text
                if stable_counter == 2:
                    clean = clean_ocr_text(current_text)
                    if len(clean) > 3:
                        already_voiced = False
                        if history:
                            for old_text in history[-2:]:
                                if SequenceMatcher(None, clean, old_text).ratio() > 0.9:
                                    already_voiced = True
                                    break
                        if not already_voiced:
                            print(f">>> {clean}")
                            voice_queue.put(clean)
                            history.append(clean)
                            last_voiced_text = clean
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