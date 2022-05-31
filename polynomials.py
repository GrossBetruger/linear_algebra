from sympy import poly
from sympy.abc import x, y

def p(degree: int, variable):
    if degree == 4:
        return poly(1 + variable + variable**2 + variable**3)
    if degree == 3:
        return poly(1 + variable + variable**2)

# print(1, -p(degree=4), p(degree=4), -2*(p(degree=4)))

print(p(degree=3, variable=(x+1))+p(degree=3, variable=(y+1)))
print(p(degree=3, variable=(x+y+1)))