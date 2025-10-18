import re
from typing import Any

from pandas import DataFrame, Series

from utils import load_csv, save_answer, log_task_done


def extract_first_name(
    name: str
) -> str | None:
    """
    Извлекает личное имя женщины из полного имени
    с титулом Miss. или Mrs.
    """
    # Ищем "Miss." или "Mrs." и берём первое слово после титула
    match: re.Match[str] | None = re.search(
        r'(?:Miss\.|Mrs\.)\s+([A-Za-z\-]+)', name
    )
    # Возвращаем имя или None, если совпадений нет
    return match.group(1) if match else None


def main() -> None:
    """
    Анализирует датасет Titanic и сохраняет результаты
    по каждому пункту задания в отдельные файлы.
    """
    # Загружаем CSV в DataFrame
    df: DataFrame = load_csv('data/titanic.csv')

    # 1. Количество мужчин и женщин
    sex_counts: Series[int] = df['Sex'].value_counts()
    # Сохраняем строку с количеством мужчин и женщин
    save_answer(
        '1.txt',
        f"{sex_counts.get('male', 0)} {sex_counts.get('female', 0)}"
    )

    # 2. Доля выживших пассажиров (%)
    # Среднее значение столбца Survived * 100
    save_answer(
        '2.txt',
        f"{df['Survived'].mean() * 100:.2f}"
    )

    # 3. Доля пассажиров 1-го класса (%)
    # Сравниваем Pclass с 1 и берём среднее * 100
    save_answer(
        '3.txt',
        f"{(df['Pclass'] == 1).mean() * 100:.2f}"
    )

    # 4. Средний и медианный возраст
    mean_age = df['Age'].mean()
    median_age = df['Age'].median()
    save_answer(
        '4.txt',
        f"{mean_age:.2f} {median_age:.2f}"
    )

    # 5. Корреляция между SibSp и Parch
    correlation = df['SibSp'].corr(df['Parch'])
    save_answer(
        '5.txt',
        f"{correlation:.2f}"
    )

    # 6. Самое популярное женское имя
    female_names: Series[Any] = df.loc[df['Sex'] == 'female', 'Name']
    most_common_name: int | str = (
        female_names
        .map(extract_first_name)  # Извлекаем личные имена
        .dropna()                 # Убираем None значения
        .astype(str)              # Приводим к строке
        .value_counts()           # Считаем частоту каждого имени
        .idxmax()                 # Берём самое частое
    )
    save_answer('6.txt', str(most_common_name))

    # Логируем, что задание выполнено
    log_task_done()


if __name__ == '__main__':
    main()
