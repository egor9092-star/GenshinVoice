import torch
import torchaudio
import soundfile as sf
import os

# --- Параметры ---
local_file_path = "model.pt" # Модель будет сохранена в той же паппе, что и скрипт
model_url = "https://models.silero.ai/models/tts/ru/v4_ru.pt" # Прямая ссылка на русскую модель

# --- Скачивание модели, если она еще не скачана ---
if not os.path.isfile(local_file_path):
    print(f"Скачиваю модель Silero v4 ({model_url})...")
    try:
        torch.hub.download_url_to_file(model_url, local_file_path)
        print("ОК! Модель успешно скачана.")
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        exit()
else:
    print("Модель уже есть в папке.")

# --- Загрузка модели из локального файла ---
print("Загружаю модель...")
model = torch.package.PackageImporter(local_file_path).load_pickle("tts_models", "model")
device = torch.device("cpu")
model.to(device)

# --- Выбор голоса и генерация речи ---
speaker = "baya"  # Доступные голоса: 'aidar', 'baya', 'kseniya', 'eugene', 'xenia'
sample_rate = 48000
text = "Привет! Это тестовый синтез речи."

print("Генерирую речь...")
audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)

# --- Сохранение в WAV-файл (можно воспроизвести любым плеером)---
sf.write("output.wav", audio, sample_rate)
print("Готово! Речь сохранена в файл output.wav")
input("Нажмите Enter, чтобы закрыть окно...")