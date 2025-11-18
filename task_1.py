"""Модуль анализа данных Titanic с извлечением статистики и женских имен."""

import re
from typing import Any, Optional, Union

import pandas as pd

from utils import load_csv, log_task_done, save_answer


def extract_first_name(
    name: str
) -> Optional[str]:
    """
    Извлекает личное женское имя из полного имени пассажирки Titanic.

    Приоритет извлечения:
        1. Имя в скобках.
        2. Имя в кавычках.
        3. Первое слово после титула (Miss, Mrs, Mme, Mlle, Lady, Countess).

    Args:
        name (str): Полное имя пассажирки.

    Returns:
        Optional[str]: Личное имя женщины или None, если извлечь не удалось.
    """
    # 1. Имя в скобках
    if "(" in name:
        inside = name.split("(", 1)[1].split(")", 1)[0]
        return inside.split()[0]

    # 2. Имя в кавычках
    quote_match = re.search(r'"([^"]+)"', name)
    if quote_match:
        return quote_match.group(1).split()[0]

    # 3. Разбор части после запятой
    if "," not in name:
        return None
    after_comma = name.split(",", 1)[1].strip()

    # Женские титулы
    titles = ["Miss", "Mrs", "Mme", "Mlle", "Lady", "Countess"]
    parts = after_comma.split()

    for i, part in enumerate(parts):
        title_clean = part.replace(".", "")
        if title_clean in titles:
            for word in parts[i + 1:]:
                if word[0].isupper():
                    return word
            return None

    return None


def main() -> None:
    """
    Выполняет анализ датасета Titanic и сохраняет результаты в файлы.

    Задачи:
        1. Подсчет мужчин и женщин.
        2. Доля выживших пассажиров.
        3. Доля пассажиров первого класса.
        4. Средний и медианный возраст.
        5. Корреляция SibSp и Parch.
        6. Самое популярное женское имя.
    """
    df: pd.DataFrame = load_csv("data/titanic.csv")

    # 1. Количество мужчин и женщин
    sex_counts: pd.Series[int] = df["Sex"].value_counts()
    save_answer(
        "1.txt",
        f"{sex_counts.get('male', 0)} {sex_counts.get('female', 0)}"
    )

    # 2. Доля выживших пассажиров (%)
    save_answer("2.txt", f"{df['Survived'].mean() * 100:.2f}")

    # 3. Доля пассажиров 1-го класса (%)
    save_answer("3.txt", f"{(df['Pclass'] == 1).mean() * 100:.2f}")

    # 4. Средний и медианный возраст
    mean_age: float = df["Age"].mean()
    median_age: float = df["Age"].median()
    save_answer("4.txt", f"{mean_age:.2f} {median_age:.2f}")

    # 5. Корреляция между SibSp и Parch
    correlation: float = df["SibSp"].corr(df["Parch"])
    save_answer("5.txt", f"{correlation:.2f}")

    # 6. Самое популярное женское имя
    female_names: pd.Series[Any] = df.loc[df["Sex"] == "female", "Name"]
    most_common_name: Union[int, str] = (
        female_names
        .map(extract_first_name)
        .dropna()
        .astype(str)
        .value_counts()
        .idxmax()
    )
    save_answer("6.txt", str(most_common_name))

    log_task_done()


if __name__ == "__main__":
    main()
