from numpy import ndarray
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier

from utils import load_csv, save_answer, log_task_done, top_n_features


def main() -> None:
    """
    Анализирует датасет Titanic с помощью DecisionTreeClassifier
    и сохраняет топ-2 наиболее важных признака.
    """
    # Загружаем CSV в DataFrame
    df: DataFrame = load_csv('data/titanic.csv')

    # 1. Определяем признаки и целевую переменную
    features: list[str] = ['Pclass', 'Fare', 'Age', 'Sex']
    df = df[features + ['Survived']]  # оставляем только нужные столбцы

    # Преобразуем пол в числовой формат
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Удаляем строки с пропусками в признаках или цели
    df = df.dropna(subset=features + ['Survived'])

    # Признаки (X) и целевая переменная (y) в виде массивов
    X: ndarray = df[features].to_numpy()  # двумерный массив признаков
    y: ndarray = df['Survived'].to_numpy()  # одномерный массив целей

    # 2. Обучаем дерево решений
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)  # обучаем модель на данных

    # 3. Определяем важности признаков
    top_features: list[str] = top_n_features(
        list(zip(features, clf.feature_importances_)), 2
    )  # выбираем 2 наиболее важных признака

    # 4. Сохраняем результат в файл
    save_answer('1.txt', ' '.join(top_features))

    # Логируем, что задание выполнено
    log_task_done()


if __name__ == '__main__':
    main()
