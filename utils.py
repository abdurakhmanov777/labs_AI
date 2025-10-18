from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import pandas as pd


def save_answer(
    filename: str | Path,
    answer: str
) -> Path:
    """
    Сохраняет строку в файл без лишнего перевода строки.

    Файл сохраняется в папку answers/<имя_скрипта>.

    Args:
        filename (str | Path): Имя файла для сохранения.
        answer (str): Содержимое для записи.

    Returns:
        Path: Путь к сохранённому файлу.
    """
    # Получаем имя вызывающего скрипта
    script_name: str = Path(inspect.stack()[1].filename).stem

    # Папка answers/<имя_скрипта>
    folder: Path = Path('answers') / script_name
    folder.mkdir(parents=True, exist_ok=True)

    # Полный путь до файла
    file_path: Path = folder / filename
    file_path.write_text(answer, encoding='utf-8')
    return file_path


def load_csv(
    *path_parts: str | Path,
    usecols: list[str] | None = None,
    header: int | None | Literal['infer'] = 'infer'
) -> pd.DataFrame:
    """
    Загружает CSV в DataFrame с опциональным выбором столбцов.

    Путь строится относительно скрипта.

    Args:
        *path_parts (str | Path): Части пути до CSV.
        usecols (list[str] | None): Выбор столбцов.
        header (int | None | 'infer'): Строка заголовка.

    Returns:
        pd.DataFrame: Загруженный DataFrame.
    """
    base: Path = Path(__file__).resolve().parent
    file_path: Path = base.joinpath(*map(str, path_parts))
    return pd.read_csv(file_path, usecols=usecols, header=header)


def top_n_features(
    importances: list[tuple[str, float]],
    n: int = 2
) -> list[str]:
    """
    Возвращает n самых важных признаков по списку (feature, importance).

    Args:
        importances (list[tuple[str, float]]): Список признаков и их важностей.
        n (int): Количество топ-признаков для возврата.

    Returns:
        list[str]: Список имен топ-n признаков.
    """
    importances.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in importances[:n]]


def log_task_done() -> None:
    """
    Логирует сообщение о завершении задания.

    Формат:
    'Задание №* завершено. Ответы сохранены в папку "answers/task_*".'
    """
    script_name: str = Path(inspect.stack()[1].filename).stem
    task_number: str = script_name.split('_')[-1]
    print(
        f'Задание №{task_number} завершено. Ответы сохранены в папку '
        f'"answers/{script_name}".'
    )
