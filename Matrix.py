import copy


# Klasa reprezentująca macierz
class Matrix:
    # Tworzenie macierzy o zadanym rozmiarze i parametrach
    def __init__(self, size, a1, a2, a3):
        self.size = size
        self.matrix = [[0.0 for _ in range(self.size)] for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.matrix[i][j] = a1
                elif i == j + 1 or j == i + 1:
                    self.matrix[i][j] = a2
                elif i == j + 2 or j == i + 2:
                    self.matrix[i][j] = a3

    # Mnożenie macierzy przez wektor
    def multiplication(self, vector):
        result = []
        for i in range(self.size):
            tmp = 0
            for j in range(len(vector)):
                tmp += self.matrix[i][j] * vector[j]

            result.append(tmp)

        return result

    # Tworzenie macierzy trójkątnej dolnej i górnej do faktoryzacji LU
    def create_LU(self):
        U = copy.deepcopy(self)
        L = Matrix(self.size, 1, 0, 0)

        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                L.matrix[j][i] = U.matrix[j][i] / U.matrix[i][i]

                for k in range(i, self.size):
                    U.matrix[j][k] = U.matrix[j][k] - L.matrix[j][i] * U.matrix[i][k]

        return L, U
