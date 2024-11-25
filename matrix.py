import random
import math
import numpy as np


class Matrix:
    def __init__(self, nrows, ncols, init="zeros"):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError()
        if init not in {"zeros", "ones", "random", "eye"}:
            raise ValueError()
        self.nrows = nrows
        self.ncols = ncols
        self.data = [[]]
        match init:
            case 'zeros':
                self.data = [[0] * self.ncols for _ in range(self.nrows)]
            case 'ones':
                self.data = [[1] * self.ncols for _ in range(self.nrows)]
            case 'random':
                self.data = [[random.uniform(0, 1) for _ in range(self.ncols)] for _ in range(self.nrows)]
            case 'eye':
                rows = []
                for index in range(ncols):
                    row = [0] * self.ncols
                    row[index] = 1
                    rows.append(row)
                self.data = rows.copy()

    def _with_data(self, data):
        """Приватный метод для обновления данных матрицы."""
        self.data = data
        self.nrows = len(data)
        self.ncols = len(data[0]) if data else 0
        return self

    @staticmethod
    def from_dict(data):
        "Десериализация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols * nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols * row + col]
        return result

    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}

    def __str__(self):
        matrix = []
        for row in self.data:
            string = '\t'.join(f"{elem:.2f}" for elem in row)
            matrix.append(string)
        return '\n'.join(matrix)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return [self.nrows, self.ncols]

    def __getitem__(self, index):
        """Получить элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        """
        if not isinstance(index, (list, tuple)):
            raise ValueError()
        if len(index) != 2:
            raise ValueError()
        if index[0] > self.nrows or index[1] > self.ncols:
            raise IndexError()
        row, col = index
        return self.data[row][col]

    def __setitem__(self, index, value):
        """Задать элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        value - Устанавливаемое значение
        """
        if not isinstance(index, (list, tuple)):
            raise ValueError()
        if len(index) != 2:
            raise ValueError()
        if index[0] > self.nrows or index[1] > self.ncols:
            raise IndexError()
        row, col = index
        self.data[row][col] = value

    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"

        if rhs.nrows != self.nrows or rhs.ncols != self.ncols:
            raise ValueError()
        result = [
            [a - b for a, b in zip(row_self, row_rhs)]
            for row_self, row_rhs in zip(self.data, rhs.data)
        ]
        result_matrix = Matrix(self.nrows, rhs.ncols, init="zeros")
        result_matrix.data = result
        return result_matrix

    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"

        if rhs.nrows != self.nrows or rhs.ncols != self.ncols:
            raise ValueError()
        result = [
            [a + b for a, b in zip(row_self, row_rhs)]
            for row_self, row_rhs in zip(self.data, rhs.data)
        ]
        result_matrix = Matrix(self.nrows, rhs.ncols, init="zeros")
        result_matrix.data = result
        return result_matrix

    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"
        if self.ncols != rhs.nrows:
            raise ValueError()
        result = [
            [
                sum(a * b for a, b in zip(self_row, rhs_col))
                for rhs_col in zip(*rhs.data)
            ]
            for self_row in self.data
        ]
        result_matrix = Matrix(self.nrows, rhs.ncols, init="zeros")
        result_matrix.data = result
        return result_matrix

    def __pow__(self, power):
        "Возвести все элементы в степень power и вернуть результат"
        result = [[elem ** power for elem in row] for row in self.data]
        result_matrix = Matrix(self.nrows, self.ncols, init="zeros")
        result_matrix.data = result
        return result_matrix

    def sum(self):
        "Вернуть сумму всех элементов матрицы"
        return sum(sum(row) for row in self.data)

    def det(self):
        if self.nrows != self.ncols:
            raise ValueError()

        def determinant(matrix):
            if len(matrix) == 1:
                return matrix[0][0]
            if len(matrix) == 2:
                return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            det = 0
            for c in range(len(matrix)):
                minor = [row[:c] + row[c + 1:] for row in matrix[1:]]
                det += ((-1) ** c) * matrix[0][c] * determinant(minor)
            return det

        return determinant(self.data)

    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        result = [[self.data[j][i] for j in range(self.nrows)] for i in range(self.ncols)]
        result_matrix = Matrix(self.nrows, self.ncols, init="zeros")
        result_matrix.data = result
        return result_matrix

    def inv(self):
        """Вычислить обратную матрицу и вернуть результат."""
        if self.nrows != self.ncols:
            raise ArithmeticError()
        det = self.det()
        if det == 0:
            raise ArithmeticError()

        if self.nrows == 2:
            a, b = self.data[0][0], self.data[0][1]
            c, d = self.data[1][0], self.data[1][1]
            result = [[d, -b], [-c, a]]
            # Делим элементы на определитель
            inverted_data = [[elem / det for elem in row] for row in result]
            return Matrix(self.nrows, self.ncols, init="zeros")._with_data(inverted_data)

        def minor(matrix, i, j):
            return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

        cofactors = []
        for r in range(self.nrows):
            cofactor_row = []
            for c in range(self.ncols):
                minor_matrix = minor(self.data, r, c)
                minor_det = Matrix(len(minor_matrix), len(minor_matrix), init="zeros")._with_data(minor_matrix).det()
                cofactor = ((-1) ** (r + c)) * minor_det
                cofactor_row.append(cofactor)
            cofactors.append(cofactor_row)

        cofactors_matrix = Matrix(self.nrows, self.ncols, init="zeros")._with_data(cofactors).transpose()

        inverted_data = [[elem / det for elem in row] for row in cofactors_matrix.data]
        return Matrix(self.nrows, self.ncols, init="zeros")._with_data(inverted_data)

    def tonumpy(self):
        "Приведение к массиву numpy"
        return np.array(self.data)


def test():
    """Тестирование класса Matrix"""

    np.random.seed(42)
    np_matrix = np.random.rand(3, 3)

    matrix = Matrix(3, 3, init="random")
    matrix.data = np_matrix.tolist()

    matrix2 = Matrix(3, 3, init="random")
    matrix2.data = np.random.rand(3, 3).tolist()
    sum_matrix = matrix + matrix2
    np_sum_matrix = np_matrix + matrix2.tonumpy()
    assert np.allclose(sum_matrix.tonumpy(), np_sum_matrix)

    diff_matrix = matrix - matrix2
    np_diff_matrix = np_matrix - matrix2.tonumpy()
    assert np.allclose(diff_matrix.tonumpy(), np_diff_matrix)

    product_matrix = matrix * matrix2
    np_product_matrix = np_matrix @ matrix2.tonumpy()
    assert np.allclose(product_matrix.tonumpy(), np_product_matrix)

    transpose_matrix = matrix.transpose()
    np_transpose_matrix = np_matrix.T
    assert np.allclose(transpose_matrix.tonumpy(), np_transpose_matrix)

    matrix_det = matrix.det()
    np_matrix_det = np.linalg.det(np_matrix)
    assert math.isclose(matrix_det, np_matrix_det)

    if matrix_det != 0:
        matrix_inv = matrix.inv()
        np_matrix_inv = np.linalg.inv(np_matrix)
        assert np.allclose(matrix_inv.tonumpy(), np_matrix_inv)

    print("OK")


test()
