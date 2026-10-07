# dfine-wrapper

D-FINE (upstream `Peterande/D-FINE`, коммит `956d170`) для детекции дефектов стали. Здесь один конфиг на
весь эксперимент, загрузка весов одной командой и исправления для Windows, RTX 50xx и прямоугольного входа.
Устроен так же, как `deim-steel`: те же `experiment.yml`, CSV, `best.pth`, графики, DDP, инференс и ONNX.
Описание самого D-FINE — в [README_DFINE.md](README_DFINE.md).

Код D-FINE распространяется под Apache-2.0 ([LICENSE](LICENSE)), коммерческое использование разрешено. Авторы
оговаривают, что веса `*_obj365` и `*_obj2coco` могут подпадать под условия датасета Objects365, и для
коммерческого использования их нельзя считать разрешёнными. Веса `*_coco` такой оговорки не имеют.

## Установка

```bash
uv sync          # .venv по pyproject.toml и uv.lock: torch 2.7.1 со сборкой CUDA 12.8 (нужна для RTX 50xx)
```

Скрипты `*.sh` сами находят `.venv` репозитория. Другой интерпретатор задаётся через
`PYTHON=/путь/к/python bash train.sh`. Без uv:
`pip install -r requirements_steel.txt` (тот же набор пакетов). Исходный `requirements.txt` D-FINE не содержит
`pycocotools`, а без него датасет не создаётся.

## 1. Веса

```bash
bash scripts/download_weights.sh                       # чекпоинт для model + pretrain из experiment.yml
bash scripts/download_weights.sh x --pretrain coco     # другой вариант: obj365 | obj2coco | coco | all
bash scripts/download_weights.sh x --backbone          # + ImageNet-бэкбон HGNetv2 (только для weights: none)
```

Всё скачивается с GitHub releases авторов (`github.com/Peterande/storage`). Уже лежащие файлы не перекачиваются.

| `pretrain` | Модели | На чём обучен | Файл |
|---|---|---|---|
| `obj365` | s, m, l, x | Objects365. Авторы советуют дообучать на своих данных с него | `weights/dfine_<m>_obj365.pth` |
| `obj2coco` | s, m, l, x | Objects365, затем COCO (лучший AP на COCO) | `weights/dfine_<m>_obj2coco.pth` (L: `_e25`) |
| `coco` | n, s, m, l, x | COCO | `weights/dfine_<m>_coco.pth` |
| бэкбон | n, s, m, l, x | ImageNet, HGNetv2 B0/B0/B2/B4/B5 | `weight/hgnetv2/PPHGNetV2_<B>_stage1.pth` |

Бэкбон нужен только для обучения без чекпоинта детектора (`weights: none`). При дообучении все веса берутся
из чекпоинта. Если GitHub недоступен: прокси `HTTPS_PROXY=http://proxy:port bash scripts/download_weights.sh`
или копия файлов с другой машины в `weights/` и `weight/hgnetv2/`.

## 2. Конфигурация: только `experiment.yml`

Редактируется один файл. Основные поля:

| Поле | Что задаёт |
|---|---|
| `name`, `output_root` | папка результатов `outputs/<name>/` |
| `model` | `n` \| `s` \| `m` \| `l` \| `x` |
| `pretrain` | стартовый чекпоинт для `weights: auto` и рецепт дообучения: `obj365` → upstream-рецепт `obj2custom`, `obj2coco` и `coco` → рецепт `custom` |
| `weights` | `auto` (чекпоинт по `pretrain`), путь к чекпоинту или `best.pth`, `none` (только ImageNet-бэкбон) |
| `data` | `format: csv` (путь к CSV, `class_merge`) или `format: coco` (пути к изображениям и json) |
| `input_size` | `[h, w]`, кратно 32. Прямоугольник (например `[128, 800]` для полос 1600×256) работает без multi-scale |
| `epochs`, `batch_size`, `workers` | расписание рецепта пересчитывается под `epochs` |
| `devices` | номера GPU, как в `nvidia-smi`. Несколько номеров включают DDP, см. [Несколько GPU](#несколько-gpu-ddp) |
| `optimizer`, `lr`, `backbone_lr_ratio`, `weight_decay` | `adamw` (рецепт D-FINE) или `sgd`; `auto` — lr рецепта, пересчитанный на batch (AdamW), или 0.01 (SGD); доля lr бэкбона из рецепта (X 0.01, L 0.05, M 0.1, S/N 0.5) |
| `best_metric` | `map50` или `map`: по этой метрике выбирается `best.pth` |

Что генератор берёт из рецепта и как пересчитывает:
- **Стадии.** В D-FINE одна граница: на ней выключаются аугментации (`RandomPhotometricDistort`,
  `RandomZoomOut`, `RandomIoUCrop`) и multi-scale, перезагружаются лучшие веса первой стадии и перезапускается
  EMA. Хвост без аугментаций занимает ту же долю эпох, что в рецепте: для 12 эпох и `obj2custom` это последние
  2 эпохи. При `epochs < 3` хвоста нет.
- **Warmup.** Warmup lr — 0.5 эпохи, а если рецепт его не задаёт (`obj2custom`), то 0. Warmup EMA — всегда
  1.5 эпохи, как в deim-steel. В `obj2custom` он равен 0, и тогда EMA с decay 0.9999 с первого шага почти не
  уходит от стартовых весов. После 1 эпохи на полосах стали mAP EMA-модели был около 0, а после 12 эпох в ней
  осталось бы около 73% стартового чекпоинта со случайными головами классов.
- **SGD.** clip 10 вместо 0.1 из рецепта AdamW, Nesterov, momentum 0.9.

### Данные в CSV

Одна строка на бокс, лишние столбцы игнорируются:

| Столбец | Значение |
|---|---|
| `image_path` | абсолютный путь к изображению (строка) |
| `instance_label` | класс (строка) |
| `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` | левый верхний и правый нижний угол, пиксели исходного изображения |
| `split` | `train`, `val` или `test`. Строки с другими значениями пропускаются с предупреждением |

- Пустые `instance_label` или bbox означают изображение без боксов: чистое изображение или неразмеченный test.
- `class_merge` объединяет классы по строковым меткам, например `{scratch_small: scratch}`. Классы нумеруются 0..K-1 в отсортированном порядке, одинаково для всех выборок.
- CSV конвертируется в COCO один раз, в `data_cache/<name>/`. Изображения не копируются. При изменении CSV конвертация повторяется.
- Старый формат `train_bboxes.csv` (`ImageId, ClassId, x_min, y_min, x_max, y_max, split` плюс папка `images`) определяется по заголовку.

`tools/steel/prepare_experiment.py` собирает из этого файла полный конфиг D-FINE
`configs/_generated/<name>.yml`. Его не нужно править руками: он перезаписывается при каждом запуске.

## 3. Обучение

```bash
bash train.sh                     # = python tools/steel/prepare_experiment.py experiment.yml --train
bash train.sh my_experiment.yml   # другой файл эксперимента
```

Результаты лежат в `outputs/<name>/`:
- `best.pth` — только лучшие веса (EMA) по `best_metric`, вместе с конфигом модели и именами классов: для инференса больше ничего не нужно. Промежуточные чекпоинты D-FINE ведёт в `outputs/<name>/.state/` и удаляет после обучения;
- `metrics.png` и `metrics.csv` — обновляются после каждой эпохи: mAP@0.5, mAP@0.5:0.95, precision, recall, F1 (точка лучшего F1 при IoU 0.5), AP@0.5 по классам, train loss, lr;
- `log.txt`, `train.log`, `summary/` (TensorBoard);
- `train_samples/`, `val_samples/` — первый батч обучения и валидации с нарисованной разметкой (upstream D-FINE), чтобы проверить аугментации и боксы.

Дообучить ещё раз с лучших весов: `weights: outputs/<name>/best.pth`.

### Несколько GPU (DDP)

```yaml
devices: [0, 1, 2, 3]   # в experiment.yml; запуск тот же: bash train.sh
```

При нескольких номерах `train.sh` запускает `torchrun --standalone --nproc_per_node=N train.py ...`: это штатный DDP
D-FINE с SyncBatchNorm, свободный порт выбирается сам. `batch_size` и `val_batch_size` задают общий batch на все
GPU, как `total_batch_size` в D-FINE: на каждой карте `batch_size / N`, число итераций за эпоху, lr и warmup те же,
что на одной карте. Поэтому batch должен делиться на N, а чтобы занять память каждой карты, его увеличивают в N раз.
`workers` считаются на каждый процесс. Лучшие веса, метрики и графики пишет только процесс rank 0.
DDP работает только на Linux. `torchrun` из Windows-сборок torch 2.7 не запускается (они собраны без libuv),
поэтому `train.sh` на Windows с несколькими `devices` остановится с сообщением об этом.

## 4. Оценка

```bash
python tools/steel/evaluate.py experiment.yml --fps                  # val: mAP, P/R/F1, AP по классам, FPS
python tools/steel/evaluate.py experiment.yml --split test --conf 0.3
```

В `outputs/<name>/` появляются:
- `<split>_predictions.csv` — предсказания в формате инференса (ниже), `confidence ≥ --conf`;
- `<split>_predictions.json` — все боксы в формате COCO;
- `eval_<split>.json` — метрики, если у выборки есть разметка. Для неразмеченного test создаются только предсказания.

FPS меряется при batch 1 и включает чтение, модель и постобработку.

## 5. Инференс

```python
from tools.steel.infer import Detector   # из корня репозитория; иначе сначала sys.path.insert(0, "<путь к dfine-wrapper>")

det = Detector("outputs/dfine_x_800x128/best.pth")             # device="cuda" по умолчанию, если есть GPU
df = det.predict("/data/new_images", conf=0.3)                 # папка
df = det.predict(["/data/a.jpg", "/data/b.jpg"])               # список путей
df = det.predict(my_df)                                        # DataFrame или CSV со столбцом image_path (+ split)
```

То же из командной строки (результат по умолчанию пишется в `predictions.csv` рядом с весами):

```bash
python tools/steel/infer.py outputs/dfine_x_800x128/best.pth /data/new_images --conf 0.3 --out preds.csv
```

`predict` возвращает pandas DataFrame в формате обучающего CSV плюс `confidence`, одна строка на бокс:

| Столбец | Значение |
|---|---|
| `image_path` | путь как на входе; относительный путь превращается в абсолютный |
| `instance_label` | имя класса, как в обучающих данных |
| `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` | углы бокса в пикселях исходного изображения, обрезаны по его границам |
| `split` | из входного CSV или DataFrame, иначе аргумент `split` (по умолчанию `test`) |
| `confidence` | уверенность модели, 0..1. В строке остаются боксы с `confidence ≥ conf` |

Изображение без боксов выше порога даёт одну строку с пустыми `instance_label`, bbox и `confidence`, как в
обучающем CSV (`keep_empty=False` или `--no-empty` такие строки убирает). Поэтому результат можно без изменений
подать обратно в `csv_to_coco`, например как псевдоразметку. Сырые результаты без порога и имён классов отдаёт
`det.detect(paths)`: по изображению выдаются `(path, (w, h), labels, boxes_xyxy, scores)`.

По умолчанию модель собирается в режиме deploy, как в upstream `tools/inference`: conv+BN энкодера слиты,
декодер обрезан до `eval_idx`. Так быстрее, но уверенности могут отличаться от валидации при обучении в
последних знаках. `Detector(..., deploy=False)` воспроизводит валидацию точно. Этим режимом пользуется
`evaluate.py`: его mAP совпадает до последнего знака с upstream-валидацией D-FINE на том же `best.pth`. С числом
в `metrics.csv` возможна разница в 4-м знаке (проверено: 0.1741 против 0.1738), потому что внутри процесса
обучения GPU выбирает другие алгоритмы свёрток.

Чекпоинтам без встроенного конфига (`last.pth`/`best_stg*.pth` upstream D-FINE, веса из `weights/`) нужен
конфиг: `Detector(ckpt, config="configs/_generated/<name>.yml")`. Для `outputs/<name>/` он находится сам. Для
COCO-весов D-FINE (`configs/dfine/dfine_hgnetv2_<m>_coco.yml`) подставляются имена 80 классов COCO.

## 6. Экспорт в ONNX

```bash
python tools/steel/export_onnx.py                      # best.pth из experiment.yml -> outputs/<name>/best.onnx
python tools/steel/export_onnx.py path/to/best.pth --out model.onnx
python tools/steel/export_onnx.py weights/dfine_x_coco.pth --config configs/dfine/dfine_hgnetv2_x_coco.yml
```

Конфиг модели и имена классов берутся из `best.pth`, поэтому экспорт совпадает с обученной моделью, даже если
`experiment.yml` потом правили: из него берётся только путь к `outputs/<name>/best.pth`. Чекпоинтам без встроенного
конфига (веса D-FINE из `weights/`, upstream-чекпоинты) нужен `--config`. После экспорта скрипт прогоняет PyTorch и
onnxruntime на нескольких изображениях (`--images` или первые из val) с batch 3 и 1 и сравнивает результаты.
Бит в бит они не совпадают: ядра CPU различаются в последних битах, а выбор top-k запросов в декодере может
поменять местами почти равные запросы. Сильнее всего это видно у слабо обученной модели: после 1 эпохи на
полосах уверенности расходились до 9e-3, а AP50 на val было 0.1756 у ONNX против 0.1740 у PyTorch. Проверка
проходит, если отсортированные уверенности отличаются меньше чем на 0.02 и не меньше 90% топ-20 детекций ONNX
есть и у PyTorch (тот же класс, IoU ≥ 0.9, уверенность ±0.01). Иначе скрипт завершается с ошибкой. Сломанный
экспорт не проходит её с большим запасом: с неверной нормализацией совпали 5% детекций.

| | Имя | Форма | Что это |
|---|---|---|---|
| вход | `images` | `float32 [N, 3, H, W]` | RGB, сжато до W×H (bilinear), /255. Нормализации у D-FINE нет (`normalize` в метаданных = null) |
| вход | `orig_target_sizes` | `int64 [N, 2]` | исходные (ширина, высота) |
| выход | `labels` | `int64 [N, 300]` | номер класса, имя — `class_names[label]` |
| выход | `boxes` | `float32 [N, 300, 4]` | x_tl, y_tl, x_br, y_br в пикселях исходного изображения |
| выход | `scores` | `float32 [N, 300]` | уверенность 0..1 |

Batch N гибкий, размер входа H×W фиксирован тем `input_size`, на котором училась модель. Sigmoid и выбор
top-300 уже внутри графа, NMS не нужен. В метаданных файла лежат `class_names`, `input_size` и `normalize`,
так что ONNX-файл самодостаточен. Его можно запускать тремя способами:
- через `Detector("best.onnx")` с тем же API: `detect` для сырых результатов, `predict` для CSV-формата;
- без кода репозитория, только onnxruntime, numpy и PIL:

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession("best.onnx", providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                                    if p in ort.get_available_providers()])
meta = sess.get_modelmeta().custom_metadata_map
names = json.loads(meta["class_names"])
h, w = json.loads(meta["input_size"])
norm = json.loads(meta["normalize"])  # [mean, std] or None

def detect(path, conf=0.5):
    img = Image.open(path).convert("RGB")
    x = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255
    if norm:
        x = (x - np.array(norm[0], np.float32)) / np.array(norm[1], np.float32)
    x = x.transpose(2, 0, 1)[None]  # [1, 3, h, w]
    size = np.array([[img.width, img.height]], dtype=np.int64)
    labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": size})
    keep = scores[0] >= conf
    return [(names[l], float(s), [round(float(v), 1) for v in b])
            for l, s, b in zip(labels[0][keep], scores[0][keep], boxes[0][keep])]

detect("bus.jpg")   # [('bus', 0.965, [3.9, 229.3, 804.9, 735.5]), ('person', 0.954, [49.5, 398.2, 244.9, 904.7]), ...]
```

- на GPU: вместо `onnxruntime` поставить `onnxruntime-gpu`, оба пакета вместе ставить нельзя. Для TensorRT см.
  [README_DFINE.md](README_DFINE.md).

## Старый способ

Upstream-запуск работает без изменений. Новые возможности включаются только ключами, которые пишет генератор
(`save_best_only`, `best_metric`, `plot_metrics`):

```bash
python train.py -c configs/dfine/custom/objects365/dfine_hgnetv2_x_obj2custom.yml --use-amp --seed=0 -t weights/dfine_x_obj365.pth
```

## Изменения относительно upstream

| Файл | Зачем |
|---|---|
| `src/solver/det_solver.py` | `save_best_only`, `best_metric`, `plot_metrics` (по умолчанию выключены); `best.pth` хранит конфиг и имена классов; `barrier()` перед перезагрузкой `best_stg1.pth` при смене стадии: в DDP остальные ранги могли читать файл, который rank 0 ещё пишет |
| `src/misc/metrics_log.py` | `metrics.csv` и `metrics.png` по эпохам |
| `src/misc/profiler_utils.py` | подсчёт FLOPs на `eval_spatial_size`: прямоугольный вход падал на квадратной заглушке |
| `src/optim/optim.py` | `SGDIgnoreBetas`: SGD, которому не мешает `betas` из базовых AdamW-конфигов |
| `src/misc/dist_utils.py` | gloo, если нет NCCL; при `WORLD_SIZE > 1` ошибка инициализации DDP больше не превращается молча в N независимых обучений |
| `src/core/yaml_utils.py` | `load_config` без общего словаря по умолчанию: второй конфиг в том же процессе смешивался с первым |
| `src/data/transforms/_transforms.py` | `PadToSize` для torchvision ≥ 0.21: `transform` / `make_params` и `F.get_size` вместо удалённого `get_spatial_size` |
| `tools/steel/*`, `scripts/download_weights.sh`, `train.sh`, `experiment.yml` | единый конфиг, запуск DDP, загрузка весов, конвертация CSV → COCO, оценка, инференс, экспорт в ONNX |

## Замечания

- **Windows: при нехватке видеопамяти драйвер молча переносит её в системную RAM вместо ошибки OOM**, и обучение замедляется в десятки раз. Batch подбирайте с запасом. Для X на 128×800 и 16 ГБ batch 20 даёт пик 12.3 ГБ по `nvidia-smi`, включая около 1.2 ГБ рабочего стола. Эпоха на 5332 изображениях идёт около 2.5 минуты на RTX 5070 Ti.
- **Прямоугольный вход:** multi-scale в D-FINE строит квадратные батчи, поэтому для прямоугольника он выключается. Проверка в `prepare_experiment.py` не даст включить его вручную. Anchors и позиционные кодировки D-FINE считает от `eval_spatial_size`; при загрузке COCO/Objects365-чекпоинта anchors для 640×640 не совпадают по форме и не грузятся, это ожидаемо.
- **Головы классов** чекпоинта (80 или 366 классов) при другом числе классов не загружаются и обучаются с нуля. Upstream-перенос голов Objects365 → COCO рассчитан только на 80 классов COCO и для своих данных не срабатывает.
- **Меньше 3 эпох:** финальная стадия без аугментаций не создаётся, иначе смена стадий попала бы на эпоху 0.
