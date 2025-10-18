import io
import pathlib
import sys
from importlib.machinery import ModuleSpec
import importlib.util
from types import ModuleType
from typing import Any, TextIO

from tqdm import tqdm


def run_task(
    module_path: pathlib.Path
) -> None:
    """
    Импортирует модуль по пути и выполняет его функцию main(),
    если она есть, перенаправляя stdout во временный буфер.
    """
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        module_path.stem,
        module_path
    )
    if spec is None or spec.loader is None:
        return

    module: ModuleType = importlib.util.module_from_spec(spec)
    old_stdout: TextIO | Any = sys.stdout
    sys.stdout = io.StringIO()
    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            module.main()
    except Exception:
        pass
    finally:
        sys.stdout = old_stdout


def main() -> None:
    base_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent
    task_files: list[pathlib.Path] = sorted(base_dir.glob('task_*.py'))
    total_tasks: int = len(task_files)

    bar_format = '{desc} {bar}| {percentage:3.0f}%'

    # Оранжевая полоса выполнения
    with tqdm(
        total=total_tasks,
        ncols=100,
        colour='#FFA500',
        bar_format=bar_format
    ) as pbar:
        for task_file in task_files:
            pbar.set_description(task_file.name)
            run_task(task_file)
            pbar.update(1)

        # Когда все выполнено — смена цвета на зеленый и описание
        pbar.set_description('Successfully')
        pbar.colour = '#01BE31'
        pbar.refresh()


if __name__ == '__main__':
    main()
