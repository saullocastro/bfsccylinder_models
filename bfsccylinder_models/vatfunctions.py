def func_VAT_P_x(x, L, thetas):
    theta_VP_1, theta_VP_2, theta_VP_3 = thetas
    x1 = 0
    x2 = L/4
    x3 = L/2
    x4 = 3*L/4
    x5 = L
    if x <= L/2:
        N1 = (x - x2)*(x - x3)/((x1 - x2)*(x1 - x3))
        N2 = (x - x1)*(x - x3)/((x2 - x1)*(x2 - x3))
        N3L = (x - x1)*(x - x2)/((x3 - x1)*(x3 - x2))
        return N1*theta_VP_1 + N2*theta_VP_2 + N3L*theta_VP_3
    else:
        N3R = (x - x4)*(x - x5)/((x3 - x4)*(x3 - x5))
        N4 = (x - x3)*(x - x5)/((x4 - x3)*(x4 - x5))
        N5 = (x - x3)*(x - x4)/((x5 - x3)*(x5 - x4))
        return N3R*theta_VP_3 + N4*theta_VP_2 + N5*theta_VP_1
