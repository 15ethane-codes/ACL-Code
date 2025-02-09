def forward_elimination(matrix, n):
    for i in range(n):
        # Find the row with the maximum element in the current column
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k

        # Swap the maximum row with the current row
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        # Make the diagonal contain all 1s and zero out below the diagonal
        for k in range(i + 1, n):
            if matrix[i][i] == 0:
                raise ValueError("Division by zero detected in matrix.")
            factor = matrix[k][i] / matrix[i][i]
            for j in range(i, n + 1):
                matrix[k][j] -= factor * matrix[i][j]


def back_substitution(matrix, n):
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if matrix[i][i] == 0:
            raise ValueError("Division by zero detected during back substitution.")
        x[i] = matrix[i][n] / matrix[i][i]
        for k in range(i - 1, -1, -1):
            matrix[k][n] -= matrix[k][i] * x[i]
    return x


def solve_system_of_equations(equations):
    n = len(equations)
    matrix = [eq[:] for eq in equations]  # Copy equations to avoid modifying the input

    forward_elimination(matrix, n)

    return back_substitution(matrix, n)


def main():
    print("Enter the number of equations (must be equal to the number of variables):")
    num_equations = int(input().strip())

    if num_equations < 2:
        print("The number of equations must be at least 2.")
        return

    print("Enter the coefficients and constants for each equation (e.g., '2 1 -1 1'):")
    equations = []
    for _ in range(num_equations):
        equation = list(map(float, input().strip().split()))
        if len(equation) != num_equations + 1:
            print(f"Each equation must have {num_equations} coefficients and 1 constant term.")
            return
        equations.append(equation)

    try:
        solutions = solve_system_of_equations(equations)
        print("The solution is:")
        for i, sol in enumerate(solutions):
            print(f"x{i + 1} = {sol:.6f}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()