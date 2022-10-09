import math
import time


# Klasa reprezentująca układ równań
class SystemOfEquations:
    # Tworzenie układu równań postaci Ax = b
    def __init__(self, matrix, b, stop_norm=10 ** -9):
        self.A = matrix
        self.b = b
        self.stop_norm = stop_norm

    # Implementacja metody Jacobiego
    def Jacobi_method(self):
        print("-------Jacobi method-------")

        x = [1 for _ in range(self.A.size)]
        tmp_x = [1 for _ in range(self.A.size)]
        iterations = 0

        start = time.time()
        try:
            while True:
                iterations += 1
                for i in range(self.A.size):
                    x_k = self.b[i]
                    for j in range(len(self.b)):
                        if j != i:
                            x_k -= self.A.matrix[i][j] * tmp_x[j]

                    x_k /= self.A.matrix[i][i]
                    x[i] = x_k

                for i in range(len(tmp_x)):
                    tmp_x[i] = x[i]

                res = self.calculate_residuum(x)

                # print() for tests
                # print("Iteration: " + str(iterations) + " norm(res) = " + str(self.Euclidean_norm(res)))
                norm = self.Euclidean_norm(res)

                if math.isinf(norm):
                    raise ValueError

                if norm <= self.stop_norm:
                    break
        except ValueError:
            print("Infinite value of the norm of residuum has been reached.")
            print("Jacobi method does not converge for these values.")

        duration = time.time() - start

        print(f"Number of iterations: {iterations}")
        print(f"Duration: {round(duration, 2)} seconds")

        return duration

    # Implementacja metody Gaussa-Seidla
    def Gauss_Seidel_method(self):
        print("-------Gauss-Seidel method-------")

        x = [1 for _ in range(self.A.size)]
        tmp_x = [1 for _ in range(self.A.size)]
        iterations = 0

        start = time.time()
        try:
            while True:
                iterations += 1
                for i in range(self.A.size):
                    x_k = self.b[i]
                    for j in range(len(self.b)):
                        if j < i:
                            x_k -= self.A.matrix[i][j] * x[j]
                        elif j >= i + 1:
                            x_k -= self.A.matrix[i][j] * tmp_x[j]

                    x_k /= self.A.matrix[i][i]
                    x[i] = x_k

                for i in range(len(tmp_x)):
                    tmp_x[i] = x[i]

                res = self.calculate_residuum(x)

                # print() for tests
                # print("Iteration: " + str(iterations) + " norm(res) = " + str(self.Euclidean_norm(res)))
                norm = self.Euclidean_norm(res)

                if math.isinf(norm):
                    raise ValueError

                if norm <= self.stop_norm:
                    break
        except ValueError:
            print("Infinite value of the norm of residuum has been reached.")
            print("Gauss-Seidel method does not converge for these values.")

        duration = time.time() - start

        print(f"Number of iterations: {iterations}")
        print(f"Duration: {round(duration, 2)} seconds")

        return duration

    # Implementacja faktoryzacji LU
    def LU_factorization(self):
        print("-------LU factorization method-------")
        start = time.time()

        L, U = self.A.create_LU()

        # Ly = b, podstawienie wprzód
        y = [None for _ in range(L.size)]
        for i in range(L.size):
            y_i = self.b[i]
            for j in range(i):
                y_i -= L.matrix[i][j] * y[j]

            y[i] = y_i

        # Ux = y, podstawienie wstecz
        x = [None for _ in range(U.size)]
        for i in reversed(range(U.size)):
            x_n = y[i]

            for j in range(i + 1, U.size):
                x_n -= U.matrix[i][j] * x[j]

            x_n /= U.matrix[i][i]
            x[i] = x_n

        duration = time.time() - start

        print(f"Duration: {round(duration, 2)} seconds")

        norm = self.Euclidean_norm(self.calculate_residuum(x))
        print(f"Norm of residuum: {norm}")

        return duration

    # Obliczanie normy euklidesowej wektora residuum
    @staticmethod
    def Euclidean_norm(vector):
        result = 0
        for x in vector:
            result += x**2

        return math.sqrt(result)

    # Obliczanie wektora residuum
    def calculate_residuum(self, x):
        res = self.A.multiplication(x)
        res = [e1 - e2 for (e1, e2) in zip(res, self.b)]

        return res
