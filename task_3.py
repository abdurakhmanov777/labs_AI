from typing import Any

from numpy import dtype, ndarray
from pandas import DataFrame
from scipy.sparse._matrix import spmatrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from utils import load_csv, log_task_done, save_answer


def main() -> None:
    """
    Анализирует датасет Wine и находит оптимальное k
    для метода k ближайших соседей до и после
    масштабирования признаков.
    """
    # Загружаем CSV (wine.data без заголовков)
    df: DataFrame = load_csv('data/wine.data', header=None)

    # Классы (y) и признаки (X)
    y: ndarray[Any] = df.iloc[:, 0].to_numpy()  # первый столбец = класс
    X: ndarray[Any] = df.iloc[:, 1:].to_numpy()  # остальные столбцы = признаки

    # Генератор разбиений для 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Поиск оптимального k до масштабирования
    best_k = 1
    best_score = 0.0
    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        # Оценка модели через cross_val_score
        scores: ndarray[
            tuple[Any, ...], dtype[Any]
        ] = cross_val_score(model, X, y, cv=kf)
        mean_score: Any = scores.mean()
        if mean_score > best_score:
            best_k: int = k
            best_score: Any = mean_score

    # Сохраняем результаты до масштабирования
    save_answer('1.txt', str(best_k))
    save_answer('2.txt', f'{best_score:.2f}')

    # Масштабируем признаки
    X_scaled: ndarray | spmatrix = scale(X)  # scale может вернуть spmatrix

    # 2. Поиск оптимального k после масштабирования
    best_k_scaled = 1
    best_score_scaled = 0.0
    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_scaled, y, cv=kf)
        mean_score = scores.mean()
        if mean_score > best_score_scaled:
            best_k_scaled: int = k
            best_score_scaled: Any = mean_score

    # Сохраняем результаты после масштабирования
    save_answer('3.txt', str(best_k_scaled))
    save_answer('4.txt', f'{best_score_scaled:.2f}')

    # Логируем завершение задания
    log_task_done()


if __name__ == '__main__':
    main()
