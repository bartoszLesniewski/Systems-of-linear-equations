import math
import matplotlib.pyplot as plt

from Matrix import Matrix
from SystemOfEquations import SystemOfEquations

# nr indeksu: 184783 - c = 8, d = 3, e = 7, f = 4
c = 8
d = 3
e = 7
f = 4


# Zadanie E - tworzenie wykresu zależności czasu trwania poszczególnych algorytmów od liczby niewiadomych N
def create_plot():
    N = [100, 500, 1000, 2000, 3000]
    time_Jacobi = []
    time_Gauss_Seidel = []
    time_LU_factorization = []

    for number in N:
        print(f"Calculations for n = {number}...")
        m = Matrix(number, 5 + e, -1, -1)
        b = [math.sin(n * (f + 1)) for n in range(m.size)]
        systemOfEquations = SystemOfEquations(m, b)

        time_Jacobi.append(systemOfEquations.Jacobi_method())
        time_Gauss_Seidel.append(systemOfEquations.Gauss_Seidel_method())
        time_LU_factorization.append(systemOfEquations.LU_factorization())

    plt.figure(figsize=(10, 5))
    plt.plot(N, time_Jacobi, label="Metoda Jacobiego")
    plt.plot(N, time_Gauss_Seidel, label="Metoda Gaussa-Seidla")
    plt.plot(N, time_LU_factorization, label="Metoda faktoryzacji LU")
    plt.title("Zależność czasu trwania poszczególnych algorytmów od liczby niewiadomych", fontweight="bold")
    plt.xlabel("Liczba niewiadomych")
    plt.ylabel("Czas trwania [s]")
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig('wykres.png', dpi=600, bbox_inches='tight')
    plt.show()


def main():
    # Zadanie A
    m1 = Matrix(9 * 100 + c * 10 + d, 5 + e, -1, -1)
    b = [math.sin(n * (f + 1)) for n in range(m1.size)]

    # Zadanie B
    print("TASK B")
    systemOfEquations1 = SystemOfEquations(m1, b)
    systemOfEquations1.Jacobi_method()
    systemOfEquations1.Gauss_Seidel_method()

    # Zadanie C
    print("\nTASK C")
    m2 = Matrix(9 * 100 + c * 10 + d, 3, -1, -1)
    systemOfEquations2 = SystemOfEquations(m2, b)
    systemOfEquations2.Jacobi_method()
    systemOfEquations2.Gauss_Seidel_method()

    # Zadanie D
    print("\nTASK D")
    systemOfEquations2.LU_factorization()

    # Zadanie E
    print("\nTASK E")
    create_plot()


if __name__ == '__main__':
    main()
