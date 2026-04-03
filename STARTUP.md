# Запуск сервисов Photo Studio

## Требования

- macOS (Apple Silicon M4 Pro, 24GB RAM)
- Python 3.13+
- ~25 GB свободного места для моделей

## Структура проекта

```
демо генератор/
└── ComfyUI/
    ├── main.py              # ComfyUI сервер (порт 8188)
    ├── web_app.py           # Photo Studio UI (порт 5050)
    ├── test_outfit_change.py   # CLI: замена одежды
    ├── test_background_merge.py # CLI: замена фона
    ├── venv/                # Python виртуальное окружение
    ├── input/               # Входные изображения
    ├── output/              # Результаты генерации
    ├── web_ui/              # Фронтенд (HTML/CSS/JS)
    └── models/
        ├── unet/            # Qwen-Image-Edit-2509-Q5_K_M.gguf
        ├── text_encoders/   # qwen_2.5_vl_7b_fp8_scaled.safetensors
        ├── vae/             # qwen_image_vae.safetensors
        └── loras/           # Qwen-Image-Lightning-4steps-V1.0.safetensors
```

## Шаг 1 — Активация окружения

```bash
cd ~/Desktop/демо\ генератор/ComfyUI
source venv/bin/activate
```

## Шаг 2 — Запуск ComfyUI (бэкенд генерации)

```bash
python main.py --force-fp32
```

- Слушает на `http://127.0.0.1:8188`
- `--force-fp32` обязателен для Apple Silicon (MPS не поддерживает FP16 для этой модели)
- Первый запуск загружает модели ~2 мин, далее ~30 сек
- **Не закрывать терминал** — сервер должен работать постоянно

Проверка: открыть `http://127.0.0.1:8188` — должен показать ComfyUI интерфейс.

## Шаг 3 — Запуск Photo Studio UI

В **новом терминале**:

```bash
cd ~/Desktop/демо\ генератор/ComfyUI
source venv/bin/activate
python web_app.py
```

- Слушает на `http://localhost:5050`
- Требует работающий ComfyUI на порту 8188
- **Не закрывать терминал**

Проверка: открыть `http://localhost:5050` — Photo Studio с двумя режимами.

## Быстрый запуск (оба сервиса)

В одном терминале:

```bash
cd ~/Desktop/демо\ генератор/ComfyUI
source venv/bin/activate
python main.py --force-fp32 &
sleep 5
python web_app.py
```

## CLI-скрипты (без веб-интерфейса)

Замена одежды:
```bash
python test_outfit_change.py \
  --person "девочка.JPG" \
  --outfit "одежда девочка.png" \
  --prompt "описание..." \
  --seed 100
```

Замена фона:
```bash
python test_background_merge.py \
  --person girl_safari.png \
  --background bg_jungle.png \
  --prompt "описание..." \
  --seed 300 \
  --prefix jungle_scene
```

> ComfyUI (шаг 2) должен быть запущен для работы любого скрипта.

## Остановка

- `Ctrl+C` в терминале с web_app.py
- `Ctrl+C` в терминале с main.py
- Или: `lsof -ti:5050 | xargs kill` и `lsof -ti:8188 | xargs kill`

## Устранение проблем

| Проблема | Решение |
|----------|---------|
| `Port 5050 in use` | `lsof -ti:5050 \| xargs kill -9` |
| `Port 8188 in use` | `lsof -ti:8188 \| xargs kill -9` |
| `MPS backend out of memory` | Закрыть другие приложения, перезапустить ComfyUI |
| Генерация зависает | Таймаут ~20 мин — MPS медленнее CUDA, это нормально |
| `ModuleNotFoundError` | `source venv/bin/activate` забыли активировать |
