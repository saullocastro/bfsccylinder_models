from .linbuck_VAFW import flinBuck_VAFW
from .koiter_cylinder_CTS import fkoiter_cylinder_CTS_circum
from .koiter_cylinder import fkoiter_cyl_SS3

def linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick, desvars,
        funcVAT, clamped=True, cg_x0=None, lobpcg_X=None, nint=4,
        num_eigvals=2):
    """
    Linear buckling analysis of a VAT cylinder with properties changing over
    the axial direction (x)

    Assumptions:
    - classical shell theory (when BFS element is used)
    - monolithic laminated properties (only one material for the whole laminate)
    - displacement controlled
    - returns the critical buckling load in consistent force units

    Parameters
    ----------
    L : float
        Cylinder length.
    R : float
        Cylinder radius.
    nx : int
        Number of nodes along axial direction (odd number recommended).
    ny : int
        Number of nodes along circumferential direction (even number
        recommended).
    E11, E22, nu12, G12 : float
        Orthotropic material properties.
    rho : float
        Density of orthotropic material.
    tow_thick : float
        FW tow thickness.
    desvars : list
        Each element of desvars is another list containing the variables
        compatible with the VAT function ``funcVAT`` being used.
    funcVAT : function
        VAT function in the form ``f(x, xmax, thetas)``, with ``x`` being the
        axial direction, ``xmax`` the maximum value of ``x`` in the domain, and
        ``thetas`` the angle values at the control points, such that the
        ``desvars`` parameter is a sequence of ``thetas``.
    clamped : bool, optional
        ``True`` if clamped, ``False`` if simply supported.
    cg_x0 : array, optional
        Initial guess for static solver.
    lobpcg_X : array, optional
        Initial guess for eigenvectors in the eigenvalue analysis.
    nint : int, optional
        Number of integration points per direction.
    num_eigvals : int, optional
        Number of eigenvalues to extract.

    Returns
    -------
    out : dict
        out['Pcr'] = critical buckling load
        out['cg_x0'] = static initial guess
        out['lobpcg_X'] = eigenvalue initial guess
        out['mass'] = mass
        out['eigvecs'] = eigenvectors

    """
    return flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick,
            desvars, funcVAT, clamped, cg_x0, lobpcg_X, nint, num_eigvals)


def koiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow, param_n,
        param_f, thetadeg_c, thetadeg_s, clamped=True, cg_x0=None, lobpcg_X=None,
        nint=4, koiter_num_modes=1):
    """
    Linear buckling analysis of a VAT cylinder with properties changing over
    the axial direction (x)

    Assumptions:
    - classical shell theory (when BFS element is used)
    - monolithic laminated properties (only one material for the whole laminate)
    - displacement controlled
    - returns the critical buckling load in consistent force units

    Parameters
    ----------


    Returns
    -------
    out : dict
        out['Pcr'] = critical buckling load
        out['cg_x0'] = static initial guess
        out['lobpcg_X'] = eigenvalue initial guess
        out['mass'] = mass
        out['eigvecs'] = eigenvectors

    """
    return fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow, param_n,
        param_f, thetadeg_c, thetadeg_s, clamped=True, cg_x0=None, lobpcg_X=None,
        nint=4, koiter_num_modes=koiter_num_modes)


def koiter_cylinder(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow, param_n,
        param_f, thetadeg_c, thetadeg_s, cg_x0=None, lobpcg_X=None,
        nint=4, num_eigvals=2, koiter_num_modes=1):
    """
    Linear buckling analysis of a VAT cylinder with properties changing over
    the axial direction (x)

    Assumptions:
    - classical shell theory (when BFS element is used)
    - monolithic laminated properties (only one material for the whole laminate)
    - displacement controlled
    - returns the critical buckling load in consistent force units

    Parameters
    ----------


    Returns
    -------
    out : dict
        out['Pcr'] = critical buckling load
        out['cg_x0'] = static initial guess
        out['lobpcg_X'] = eigenvalue initial guess
        out['mass'] = mass
        out['eigvecs'] = eigenvectors

    """
    return None
    #return fkoiter_cyl_SS3(L, R, rCTS, nxt, ny, prop, cg_x0,
            #lobpcg_X, nint, num_eigvals, koiter_num_modes)
