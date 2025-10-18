from typing import Any, Optional

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

from utils import load_csv, log_task_done, save_answer


def main() -> None:
    """
    Анализирует датасет Boston с помощью KNeighborsRegressor
    и определяет оптимальное значение параметра p в метрике
    Минковского.
    """
    # Загружаем CSV в DataFrame
    df: DataFrame = load_csv('data/boston.csv')

    # Признаки (X) и целевая переменная (y)
    X: ndarray = df.drop(columns=['MEDV']).to_numpy()
    y: ndarray = df['MEDV'].to_numpy()

    # Масштабируем признаки, приводим к ndarray
    X_scaled: ndarray = np.array(scale(X))

    # Генератор разбиений для 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Сетка значений параметра p (1..10, 200 точек)
    p_values: ndarray = np.linspace(1, 10, 200)

    best_p: Optional[float] = None
    best_score: float = -np.inf  # scoring = neg_mean_squared_error

    # Перебор всех p для поиска оптимального
    for p in p_values:
        model = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            p=p
        )
        # Средний отрицательный MSE на кросс-валидации
        score: Any = cross_val_score(
            model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error'
        ).mean()
        if score > best_score:
            best_score, best_p = score, float(p)

    # Сохраняем результаты
    save_answer('1.txt', f'{best_p:.1f}')
    save_answer('2.txt', f'{abs(best_score):.1f}')

    # Логируем завершение задания
    log_task_done()


if __name__ == '__main__':
    main()
