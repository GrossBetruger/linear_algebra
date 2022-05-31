from fractions import Fraction

import numpy as np

import matrix_manipulation
from matrix_manipulation import do_row_operation, ElementaryOperation, do_series_of_operations, free_text_reduce


def test_matrix_factor_row():
    A = np.array([[1, 2], [2, 1]])

    first_row_factored_by_two = do_row_operation(
        matrix=A, left_row=1, op=ElementaryOperation.Factor, factor=2
    )
    expected = np.array([[2, 4], [2, 1]])

    assert np.array_equal(first_row_factored_by_two, expected)

    second_row_factored_by_three = do_row_operation(
        matrix=A, left_row=2, op=ElementaryOperation.Factor, factor=3
    )
    expected = np.array([[1, 2], [6, 3]])

    assert np.array_equal(second_row_factored_by_three, expected)


def test_swap_rows():
    A = np.array([[1, 2], [2, 1]])

    expected = np.array([[2, 1], [1, 2]])

    swap_one_two = do_row_operation(
        matrix=A, left_row=1, op=ElementaryOperation.Swap, right_row=2
    )

    assert np.array_equal(swap_one_two, expected)

    A = np.array(
        [
            [1, 1, 5],
            [2, 1, 6],
            [2, 1, 7],
        ]
    )

    expected = np.array(
        [
            [1, 1, 5],
            [2, 1, 7],
            [2, 1, 6],
        ]
    )

    swap_two_three = do_row_operation(
        matrix=A, left_row=2, right_row=3, op=ElementaryOperation.Swap
    )

    assert np.array_equal(swap_two_three, expected)


def test_row_addition():
    A = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    expected = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [8, 10, 12],
        ]
    )

    add_1_r1_to_r3 = do_row_operation(
        matrix=A, left_row=3, right_row=1, factor=1, op=ElementaryOperation.Add
    )

    assert np.array_equal(expected, add_1_r1_to_r3)

    add_2_r1_to_r3 = do_row_operation(
        matrix=A, left_row=3, right_row=1, factor=2, op=ElementaryOperation.Add
    )

    expected = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [9, 12, 15],
        ]
    )

    assert np.array_equal(expected, add_2_r1_to_r3)

    subtract_1_and_half_r2_from_r1 = do_row_operation(
        matrix=A,
        left_row=1,
        right_row=2,
        factor=Fraction(-3, 2),
        op=ElementaryOperation.Add,
    )
    expected = np.array(
        [
            [-5, -5.5, -6],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    assert np.array_equal(expected, subtract_1_and_half_r2_from_r1)


def test_series_of_row_operations():
    A = np.array([
        [1, 0, 2, 3],
        [2, -1, 3, 6],
        [1, 4, 4, 0],
    ])

    operations_to_canonical = [
        (2, 3, -2, ElementaryOperation.Add),
        (3, 1, -1, ElementaryOperation.Add),
        (3, 2, Fraction(4, 9), ElementaryOperation.Add),
        (1, 3, 9, ElementaryOperation.Add),
        (3, None, Fraction(-9, 2), ElementaryOperation.Factor),
        (2, 3, 5, ElementaryOperation.Add),
        (2, None, Fraction(-1, 9), ElementaryOperation.Factor),
    ]

    expected = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, Fraction(-3, 2)],
        [0, 0, 1, Fraction(3, 2)]
    ])

    assert np.array_equal(expected, do_series_of_operations(A, operations=operations_to_canonical))


def test_free_text_reduce_1():
    expected_latex = '\\begin{bmatrix}\n  1 & 1 & 4 & 1 & 0\\\\\n  2 & 1 & 1 & 2 & 0\\\\\n\\end{bmatrix}\\overset{R_{2}\\rightarrow{}R_{2}-2R_{1}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 1 & 4 & 1 & 0\\\\\n  0 & 4 & 3 & 0 & 0\\\\\n\\end{bmatrix}\\overset{R_{1}\\rightarrow{}R_{1}+R_{2}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 0 & 2 & 1 & 0\\\\\n  0 & 4 & 3 & 0 & 0\\\\\n\\end{bmatrix}\\overset{R_{2}\\rightarrow{}-R_{2}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 0 & 2 & 1 & 0\\\\\n  0 & 1 & 2 & 0 & 0\\\\\n\\end{bmatrix}'

    m = np.array([
        [1, 1, -1, 1, 0],
        [2, 1, 1, -3, 0],
    ])

    commands = [
        "r2=r2-2r1",
        "r1=r1+r2",
        "r2=-r2",
    ]

    matrix_reduction_latex = free_text_reduce(mat=m, cmds=commands, finite_field="Z5", return_latex=True)
    assert matrix_reduction_latex == expected_latex
    expected_matrix = np.array([[1, 0, 2, 1, 0],
                                [0, 1, 2, 0, 0]])
    reduced_matrix = free_text_reduce(mat=m, cmds=commands, finite_field="Z5")
    assert np.allclose(reduced_matrix, expected_matrix)


def test_free_text_reduce2():
    matrix = [np.array([1, 1, 2]), np.array([2, 2, 1])]
    reduce_ops = [
        "r2=r2-2r1",
        "r2=-1/3r2",
        "r1=r1-2r2",
    ]
    expected = np.array(
        [[1, 1, 0],
         [0, 0, 1]]
    )
    reduced = free_text_reduce(np.array(matrix),
                               cmds=reduce_ops)
    assert np.array_equal(expected, reduced)

    reduction_latex = free_text_reduce(np.array(matrix),
                                       cmds=reduce_ops, return_latex=True)
    assert reduction_latex == '\\begin{bmatrix}\n  1 & 1 & 2\\\\\n  2 & 2 & 1\\\\\n\\end{bmatrix}\\overset{R_{2}\\rightarrow{}R_{2}-2R_{1}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 1 & 2\\\\\n  0 & 0 & -3\\\\\n\\end{bmatrix}\\overset{R_{2}\\rightarrow{}-1/3R_{2}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 1 & 2\\\\\n  0 & 0 & 1\\\\\n\\end{bmatrix}\\overset{R_{1}\\rightarrow{}R_{1}-2R_{2}}{\\longrightarrow}\\begin{bmatrix}\n  1 & 1 & 0\\\\\n  0 & 0 & 1\\\\\n\\end{bmatrix}'


def test_matrix_flatten():
    U = np.array([
        [[1, 0], [3, 0]],  # u1
        [[2, 1], [3, 1]],  # u2
        [[1, 4], [1, 4]]  # u3
    ])

    u1, u2, u3 = U

    assert np.array_equal(matrix_manipulation.flatten_matrix(u1), np.array([1, 0, 3, 0]))
    assert np.array_equal(matrix_manipulation.flatten_matrix(u2), np.array([2, 1, 3, 1]))
    assert np.array_equal(matrix_manipulation.flatten_matrix(u3), np.array([1, 4, 1, 4]))

