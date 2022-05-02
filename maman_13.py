import re
from typing import List, Optional

import mpmath.libmp
import numpy as np
import frozenlist
from sympy import Matrix

#
# # question 3 verification
#
# # a.
from sympy.matrices.common import NonInvertibleMatrixError

from matrix_manipulation import bmatrix, do_row_operation, ElementaryOperation, free_text_reduce

#
# v1 = np.array([0, 1])
# v2 = np.array([-1, 1])
# v3 = np.array([1, 1])
#
# print("span v1, v2:", Matrix(np.array([v1, v2])).rref())
# print("span v1, v3:", Matrix(np.array([v1, v3])).rref())
# print("span v2, v3:", Matrix(np.array([v2, v3])).rref())
#
# print(f"v1={bmatrix(v1)}")
# print(f"v2={bmatrix(v2)}")
# print(f"v3={bmatrix(v3)}")
#
# print(bmatrix(np.array([v1, v2])))
#
#
# # question 4 verification:
#
# K = Matrix(
#     np.array([
#         [1, 1, -1, 1, 0],
#         [2, 1, 1, -3, 0],
#     ])
# )
#
# print("K reduced", K.rref())
#
#
# def element_of_K(z: int, t: int):
#     if z not in [0, 1, 2, 3, 4] or t not in [0, 1, 2, 3, 4]:
#         raise ValueError(f'{t, z} not from Z_5')
#     return np.array([(-2*z-t)%5, (-2*z)%5, z, t])
#
#
# print("elements of K:")
# elements_of_k = []
# for z in range(5):
#     for t in range(5):
#         elem = element_of_K(z, t)
#         elements_of_k.append(elem)
#         print(elem)
#
#
# for el1 in elements_of_k:
#     for el2 in elements_of_k:
#         assert list(((el1 + el2) % 5)) in [list(l) for l in elements_of_k]
#
# for _lambda in [0, 1, 2, 3, 4]:
#     for elem in elements_of_k:
#         assert list(((_lambda*elem)%5)) in [list(l) for l in elements_of_k]




m = np.array([
        [1, 1, -1, 1, 0],
        [2, 1, 1, -3, 0],
    ])

commands = [
    "r2=r2-2r1",
    "r1=r1+r2",
    "r2=-r2",
]

print();print()

free_text_reduce(mat=m, cmds=commands, finite_field="Z5")


#question 5 verification

def finite_field_enum_f2_enum(field: int):
    if not mpmath.libmp.isprime(field):
        raise ValueError(f'field: {field} is not prime')

    for i in range(field):
        for j in range(field):
            yield (i, j)


def finite_field_lambda_enum(field: int):
    if not mpmath.libmp.isprime(field):
        raise ValueError(f'field: {field} is not prime')

    return list(range(field))


# make sure every non zero vector in an F2 vector space over a finite field Zn has exactly
# n linearly dependant vectors (including itself and the zero vector)
for ff in [2, 3, 5, 7, 11, 17]:
    field = ff
    f2_elements = list(finite_field_enum_f2_enum(field))
    for elem in f2_elements[1:]:
        deps = []
        λ: int
        for λ in finite_field_lambda_enum(field):
            linear_comb = tuple((λ * np.array(elem))%field)
            if linear_comb in f2_elements:
                deps.append(linear_comb)
        assert len(deps) == field


def count_m2_square_invertibles(finite_field: int) -> tuple[int, int, int]:
    if not mpmath.libmp.isprime(finite_field):
        raise ValueError(f'field: {finite_field} is not prime')
    count = int()
    all_matrices = int()
    singular = int()
    for a in finite_field_lambda_enum(finite_field):
        for b in finite_field_lambda_enum(finite_field):
            for c in finite_field_lambda_enum(finite_field):
                for d in finite_field_lambda_enum(finite_field):
                    all_matrices += 1
                    matrix = np.array([
                        [a, b],
                        [c, d],
                    ])
                    if not np.allclose(round(np.linalg.det(matrix)) % field, 0):
                        count += 1

    return count, all_matrices, singular


for field in [2, 3, 5, 7, 11]:
    formula = (field**2-1) * (field**2 - field)
    found, total, singular = count_m2_square_invertibles(field)
    assert total == field**4
    print(f"2X2(Z{field}) num invs:", found, "calculated:", formula, "num matrices:", total,
          "diff:", formula-found, "singular:", total-found, singular)


def random_linear_combination(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    random_lambda = np.random.randint(-30, 30)# *  np.random.rand()
    random_mu =  np.random.randint(-30, 30) #* np.random.rand()
    return random_lambda * a + random_mu * b

W_generating_set = [np.array([1, 3, 4]), np.array([2, 5, 1])]
U_generating_set = [np.array([1, 1, 2]), np.array([2, 2, 1])]

print("U:")
U = []
for _ in range(100000):
    rand_u = random_linear_combination(U_generating_set[0], U_generating_set[1])
    rand_u = frozenlist.FrozenList(rand_u)
    rand_u.freeze()
    U.append(rand_u)
    # print(random_linear_combination(U_generating_set[0], U_generating_set[1]))

print()

print("W:")
W = []
for _ in range(100000):
    rand_w = random_linear_combination(W_generating_set[0], W_generating_set[1])
    rand_w = frozenlist.FrozenList(rand_w)
    assert rand_w[0]*-17 + rand_w[1] * 7 == rand_w[2]
    rand_w.freeze()
    W.append(rand_w)
    # print(rand_w)

print()
print("intersection W, U")

for v in set(w for w in W) & set(u for u in U):
    print(v)


u_reduced = free_text_reduce(np.array(U_generating_set),
                       cmds=[
                           "r2=r2-2r1",
                           "r2=-1/3r2",
                           "r1=r1-2r2",
                       ]
                       )
w_reduced = free_text_reduce(np.array(W_generating_set),
                             cmds=[
                                 "r2=r2-2r1",
                                 "r1=r1+3r2",
                                 "r2=-r2"
                             ])
print(w_reduced)
# f = frozenlist.FrozenList([1, 2, 3])
# f.freeze()
# print(list(f))