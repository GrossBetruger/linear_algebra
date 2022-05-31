import numpy as np

from matrix_manipulation import matrix_pprint as pp

# question 1.

#a. not true matrix multiplication of upper diagonals not associative (not a field)

#counter example:


upper1 = np.array(
    [
        [3, 2],
        [0, 1],
    ]
)

upper2 = np.array(
    [
        [1, 2],
        [0, 3],
    ]
)

upper_12 = upper1 @ upper2
upper_21 = upper2 @ upper1

print(pp(upper_12))
print()
print(pp(upper_21))