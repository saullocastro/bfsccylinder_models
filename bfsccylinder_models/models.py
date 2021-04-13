from .models_core import flinearBucklingVATCylinder_x

def linearBucklingVATCylinder_x(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick, desvars,
            clamped=True, cg_x0=None, lobpcg_X=None, nint=4, num_eigvals=2):
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
        Cylinder length
    R : float
        Cylinder radius
    nx : int
        Number of nodes along axial direction (odd number recommended)
    ny : int
        Number of nodes along circumferential direction (odd number recommended)
    E11, E22, nu12, G12 : float
        Orthotropic material properties
    rho : float
        Density of orthotropic material
    tow_thick : float
        FW tow thickness
    desvars : list
        Each element of desvars is another list containing the variables
        compatible with the VAT function being used. It is assumed that
    clamped : bool
        True if clamped, False if simply supported
    cg_x0 : array, optional
        Initial guess for static solver
    lobpcg_X : array, optional
        Initial guess for eigenvectors in the eigenvalue analysis
    nint : int, optional
        Number of integration points per direction
    num_eigvals : int, optional
        Number of eigenvalues to extract

    Returns
    -------
    out : dict
        out['Pcr'] = critical buckling load
        out['cg_x0'] = static initial guess
        out['lobpcg_X'] = eigenvalue initial guess
        out['mass'] = mass
        out['eigvecs'] = eigenvectors

    """
    return flinearBucklingVATCylinder_x(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick, desvars,
            clamped, cg_x0, lobpcg_X, nint, num_eigvals)
