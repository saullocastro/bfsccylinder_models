import sympy
sympy.var('n, f')
m = sympy.Matrix([[n, n+1],
                  [n, -f*(n+1)]])
minv = sympy.siplify(sympy.simplify(m.inv()))
print(minv)
