def matrix_generator():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = []

    for i in range(rows):
        row_input = input(f"Enter row {i + 1} (space-separated values): ")
        row = row_input.split()
        if len(row) != cols:
            print("Row length does not match the number of columns. Try again.")
            return None
        try:
            row = [float(value) for value in row]
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return None
        matrix.append(row)

    print("Entered Matrix:")
    for row in matrix:
        print(row)
    return matrix


def matrix_addition(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        print("Matrices must be the same size for addition.")
        return None
    result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    print("Resultant Matrix after Addition:")
    for row in result:
        print(row)
    return result


def matrix_subtraction(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        print("Matrices must be the same size for subtraction.")
        return None
    result = [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    print("Resultant Matrix after Subtraction:")
    for row in result:
        print(row)
    return result


def scalar_multiplication(matrix, scalar):
    result = [[matrix[i][j] * scalar for j in range(len(matrix[i]))] for i in range(len(matrix))]
    print(f"Resultant Matrix after Scalar Multiplication by {scalar}:")
    for row in result:
        print(row)
    return result


def matrix_multiplication(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        print("Number of columns of the first matrix must be equal to the number of rows of the second matrix.")
        return None
    result = []
    for i in range(len(matrix1)):
        result_row = []
        for j in range(len(matrix2[0])):
            sum_product = 0
            for k in range(len(matrix2)):
                sum_product += matrix1[i][k] * matrix2[k][j]
            result_row.append(sum_product)
        result.append(result_row)
    print("Resultant Matrix after Multiplication:")
    for row in result:
        print(row)
    return result


if __name__ == "__main__":
    print("Matrix Generator")
    matrix1 = matrix_generator()
    matrix2 = matrix_generator()

    if matrix1 is not None and matrix2 is not None:
        print("\nMatrix Addition:")
        matrix_addition(matrix1, matrix2)

        print("\nMatrix Subtraction:")
        matrix_subtraction(matrix1, matrix2)

        scalar = None
        while scalar is None:
            scalar_input = input("\nEnter a scalar value for multiplication: ")
            try:
                scalar = float(scalar_input)
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

        print("\nScalar Multiplication:")
        scalar_multiplication(matrix1, scalar)

        print("\nMatrix Multiplication:")
        matrix_multiplication(matrix1, matrix2)

