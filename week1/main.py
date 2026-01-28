# write a Python code to complete the following task
# use a random way to generate a two dimensional matrix for grid processing. The
# matrix is composed of 10 rows and 10 columns, where the entries of the matrix is
# either 0 or 1

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
def matrixGeneration(matrix_level):
    matrix = list()
    for row in [[random.randint(0,1) for _ in range(matrix_level)] for _ in range(matrix_level)]:
        matrix.append(row)

    print(matrix)


# matrix = [[random.randint(0, 1) for _ in range(matrix_level)] for _ in range(matrix_level)]
# for row in matrix:
#     print(row)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    matrixGeneration(10)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
