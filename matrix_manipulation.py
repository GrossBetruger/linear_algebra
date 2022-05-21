import codecs
import re
from copy import deepcopy
from enum import Enum, auto
from typing import Optional, Union, List, Tuple
from fractions import Fraction

import numpy as np
import pandas as pd


class ElementaryOperation(Enum):
    Add = auto()
    Factor = auto()
    Swap = auto()


def do_row_operation(
        matrix: np.ndarray,
        left_row: int,
        op: ElementaryOperation,
        right_row: Optional[int] = None,
        factor: Optional[Union[int, Fraction]] = None,
        support_fractions: Optional[bool] = True
) -> np.ndarray:
    if support_fractions is True:
        matrix = deepcopy(matrix + Fraction())
    # move from base1 to base0
    left_row -= 1
    right_row = right_row - 1 if right_row is not None else None

    if op is ElementaryOperation.Swap:
        matrix[[left_row, right_row]] = matrix[[right_row, left_row]]
        return matrix

    if op is ElementaryOperation.Factor:
        matrix[left_row] = factor * matrix[left_row]
        return matrix

    if op is ElementaryOperation.Add:
        matrix[left_row] += factor * matrix[right_row]
        return matrix

    raise ValueError(f'unknown operation: {op}')


def do_series_of_operations(matrix, operations: List[Tuple]):
    matrix = matrix.copy()
    for args in operations:
        left, right, factor, op = args
        matrix = do_row_operation(matrix=matrix, left_row=left, right_row=right, factor=factor, op=op)
    return matrix


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def matrix_pprint(matrix: np.ndarray) -> str:
    return "\n".join(pd.DataFrame(matrix).to_string(index=False).splitlines()[1:])


def test_symmetry(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, matrix.transpose())


def test_anti_symmetry(matrix: np.ndarray) -> bool:
    return np.allclose(-matrix, matrix.transpose())


def parse_row_command(cmd: str):
    def parse_factor(text: str) -> Union[int, Fraction]:
        fraction_factor = re.search("\d+/\d+", text)
        if fraction_factor:
            numerator, denominator = fraction_factor.group(0).split("/")
            factor = Fraction(int(numerator), int(denominator))
            return factor

        factor = re.search("([\-\+])(\\d+)", text)
        if factor:
            return int(factor.group(2))
        # implicit factor of 1
        return 1

    def parse_rest(text: str):
        right_row = re.search("r\d$", text).group(0)
        op = re.search("[\\-\+]", text)
        if op:
            op = op.group(0)
        else:
            op = None
        # factor = re.search("([\-\+])(\\d+)", text)
        # fraction_factor = re.search("\d+/\d+", text)
        # if factor:
        #     factor = int(factor.group(2))
        # else: # implicit factor of 1
        #     factor = 1
        # if fraction_factor:
        #     numerator, denominator = fraction_factor.group(0).split("/")
        #     factor = Fraction(numerator, denominator)
        factor = parse_factor(text)
        return right_row, op, factor

    left_row, rest = cmd.split("=")
    right_row, op, factor = parse_rest(rest)
    if op == "-":
        factor = -factor
    if op in ["+", "-"]:
        op = ElementaryOperation.Add

    # print("left", left_row, "right", right_row, "op", op, "factor", factor)
    if left_row == right_row:
        op = ElementaryOperation.Factor

    return left_row, right_row, op, factor


def free_text_row_operation(m: np.array, cmd: str) -> np.ndarray:
    left_row, right_row, op, factor = parse_row_command(cmd)
    print(left_row, right_row, op, factor)
    m = do_row_operation(
        matrix=m,
        left_row=int(left_row[1]),
        right_row=int(right_row[1]),
        factor=factor,
        op=op,
        support_fractions=False,
    )
    return m


def latex_row_operation_notation(row_operation_text):
    left_row, right_row, _, factor = parse_row_command(row_operation_text)
    if factor > 0:
        if factor == 1:
            factor = "+"
        else:
            factor = f"+{factor}"
    elif factor == -1:
        factor = "-"

    left_row = left_row[1:]
    right_row = right_row[1:]
    if left_row == right_row:
        return rf"\overset{{R_{{{left_row}}}\rightarrow{{}}{factor}R_{{{right_row}}}}}{{\longrightarrow}}"
    return rf"\overset{{R_{{{left_row}}}\rightarrow{{}}R_{{{left_row}}}{factor}R_{{{right_row}}}}}{{\longrightarrow}}"


def free_text_reduce(mat: np.ndarray, cmds: List[str], finite_field: Optional[str] = None,
                     return_latex: Optional[bool] = False) -> np.ndarray:
    if finite_field is not None:
        prime = int(re.search("(Z|z)(\d)", finite_field).group(2))
        mat = mat % prime

    latex_print = bmatrix(mat)
    for cmd in cmds:
        # print(cmd)
        mat = free_text_row_operation(mat, cmd)
        # print(mat)
        if finite_field is not None:
            prime = int(re.search("(Z|z)(\d)", finite_field).group(2))
            mat = mat % prime

        latex_print += latex_row_operation_notation(cmd)
        latex_print += bmatrix(mat)

    print(latex_print)
    if return_latex is True:
        return latex_print
    return mat


def flatten_matrix(mat: np.ndarray) -> np.ndarray:
    return np.array(list(np.matrix(mat).flatten().flat))
