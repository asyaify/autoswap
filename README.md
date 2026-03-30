# autoswap

Пайплайн для замены одежды на детских фото, переноса позы и замены фона.

Стек:
- SAM2.1 для выделения человека и области одежды
- Эвристика по bbox человека для стартовой области одежды
- SDXL inpainting + ControlNet OpenPose для замены одежды и переноса позы
- Композитинг по маске для смены фона

Ограничения:
- Проект рассчитан на нейтральную, бытовую обработку фото.
- Для работы генерации нужен GPU с CUDA и доступ к весам моделей Hugging Face.
- Маскирование использует `facebook/sam2.1-hiera-large` по умолчанию и может скачать веса с Hugging Face автоматически.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Модели

По умолчанию используются:
- `facebook/sam2.1-hiera-large`
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- `thibaud/controlnet-openpose-sdxl-1.0`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `lllyasviel/Annotators`

Если нужен локальный SAM2, можно передать `--sam-config` и `--sam-checkpoint`.

## Быстрый запуск

```bash
PYTHONPATH=src python -m autoswap.cli run \
	--input assets/child.jpg \
	--pose-ref assets/pose.jpg \
	--output-dir outputs/run01 \
	--clothing-prompt "bright yellow rain jacket, denim overalls, realistic children's outfit" \
	--background-prompt "sunny playground, natural light, shallow depth of field" \
	--garment-scope upper
```

## Что делает пайплайн

1. Ищет основного человека через SAM2 automatic mask generator.
2. Строит маску одежды:
	 - либо из `--cloth-box`
	 - либо эвристически из bounding box человека
3. Извлекает позу из `--pose-ref` через OpenPose.
4. Генерирует новую одежду и новую позу через SDXL inpainting + ControlNet.
5. Повторно сегментирует итоговый кадр и подменяет фон.

## Поэтапный запуск

Можно выполнять не все сразу, а по стадиям.

Только маски:

```bash
PYTHONPATH=src python -m autoswap.cli mask \
	--input assets/child.jpg \
	--output-dir outputs/mask01
```

Маски + перенос позы + новая одежда:

```bash
PYTHONPATH=src python -m autoswap.cli swap \
	--input assets/child.jpg \
	--pose-ref assets/pose.jpg \
	--output-dir outputs/swap01 \
	--clothing-prompt "green hoodie, cargo pants, realistic children's outfit"
```

Только замена фона:

```bash
PYTHONPATH=src python -m autoswap.cli background \
	--input outputs/swap01/04_swapped.png \
	--output-dir outputs/bg01 \
	--background-prompt "bright kindergarten classroom, natural daylight"
```

## Основные аргументы

```bash
PYTHONPATH=src python -m autoswap.cli --help
```

Ключевые параметры:
- `--clothing-prompt` описание новой одежды
- `--garment-scope` `upper`, `lower` или `full`
- `--cloth-box` ручная рамка области одежды в формате `x1,y1,x2,y2`
- `--pose-ref` референс позы
- `--background-prompt` текст для генерации нового фона
- `--background-image` готовый фон вместо генерации
- `--seed` фиксирует воспроизводимость

## Выходные файлы

В директорию результата сохраняются:
- `01_subject_mask.png`
- `02_clothing_mask.png`
- `03_pose_condition.png`
- `04_swapped.png`
- `05_final.png`

## Замечания по качеству

- Если одежда захватывает кожу или лицо, лучше передать точный `--cloth-box`.
- Если нужен более стабильный силуэт, используйте референс позы с похожим кадрированием.
- Для каталожного качества обычно нужен второй проход ретуши или ручная правка маски.