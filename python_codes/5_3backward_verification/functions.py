import numpy as np

## Arc coordinates
def arc_points(xc, yc, R, theta_start, theta_end, num):
    theta = np.linspace(theta_start, theta_end, num)
    x = xc + R * np.cos(theta)
    y = yc + R * np.sin(theta)
    return np.column_stack([x, y])

## Charge point computation
def charge_point(z_jw, r_j):
    # Charge point distribution along C4 periphery, also clock-wise
        # Average distance of adjacent points, H_{k} = 1/2 * (|z_{k+1} - z_{k}| + |z_{k} - z_{k-1}|)
        # Using entire shift, np.roll
        z_jw_prev = np.roll(z_jw,1)
        z_jw_next= np.roll(z_jw,-1)
        z_jw_dist_back = np.abs(z_jw - z_jw_prev)
        z_jw_dist_forw = np.abs(z_jw_next - z_jw)
        z_jw_H = 0.5 * (z_jw_dist_back + z_jw_dist_forw)
        # Outward direction of adjacent points, Theta_{k} = arg(z_{k+1} - z_{k-1})
        # Illustration
        # z_{k-1} ————> z_{k} ————> z_{k+1}
        #                |
        #                |
        #                ↓ 
        #         Outward direciton
        z_jw_vect = z_jw_next - z_jw_prev
        z_jw_angl = np.angle(z_jw_vect)
        z_jw_theta = z_jw_angl - np.pi/2.
        z_jw_exp_itheta = np.exp(1j * z_jw_theta)
        # Charge points
        X_jw = z_jw + r_j * z_jw_H * z_jw_exp_itheta
        return X_jw

def forward_mapping_solution(z_jw, X_jw, j, N, z_jwc):
    # Construct coefficient matrix and constant vector
    # K_{j}, Quantity of collocation points and charge points
    Kj = z_jw.size
    if (j == N):
        # N = K_{j} 
        CjN = Kj
        ACj = np.zeros((CjN, CjN), dtype = np.float64)
        bCj = np.zeros(CjN, dtype = np.float64)
        # | x x x ... x x x |
        # | x x x ... x x x |
        # | x x x ... x x x |
        # | . . . ... . . . |
        # | x x x ... x x x |
        # | x x x ... x x x |
        # Preparation: array dimension extension of collocation points and charge points
        z_jwl_matrix = z_jw[:, np.newaxis]
        X_jwk_matrix = X_jw[np.newaxis, :]
        # Distance matrix first, using broadcast technique
        Cj_distance_matrix = np.abs(z_jwl_matrix - X_jwk_matrix)
        # Log(distance)
        Cj_log_distance_matrix = np.log(Cj_distance_matrix)
        # Fill all x items
        ACj = Cj_log_distance_matrix
        # Construct constant vector
        bCj = -np.log(np.abs(z_jw - z_jwc))
        # Compute 2-norm condition number
        kappa2Cj = np.linalg.cond(ACj)
        # print("  Exterior periphery, no replacement necessary!")
        print(f"  Coefficient matrix 2-norm condition number for C{j} = {kappa2Cj:.4e}")
        # Linear system solution
        try:
            XCj = np.linalg.solve(ACj,bCj)
        except np.linalg.LinAlgError as err:
            print("Ill-conditioned matrix, solution failure!")
        # Solution separation, Q_{k} and Gamma
        Gamma_j = 0.
        QCj = XCj
    else:
        # N = K_{j} + 1
        CjN = Kj + 1
        ACj = np.zeros((CjN, CjN), dtype = np.float64)
        bCj = np.zeros(CjN, dtype = np.float64)
        # Construct coefficient matrix in three steps
        # | x x x ... x x x y|
        # | x x x ... x x x y|
        # | x x x ... x x x y|
        # | z z z ... z z z 0|
        # First step, construct x items, which share similar pattern of logarithmic distance between z_{l} and X_{k}
        # Second step, construct y items, whose values are all unity
        # Third step, construct z items, whose values are all unity
        # Preparation: array dimension extension of collocation points and charge points
        z_jwl_matrix = z_jw[:, np.newaxis]
        X_jwk_matrix = X_jw[np.newaxis, :]
        # Distance matrix first, using broadcast technique
        Cj_distance_matrix = np.abs(z_jwl_matrix - X_jwk_matrix)
        # Log(distance)
        Cj_log_distance_matrix = np.log(Cj_distance_matrix)
        # Fill all x items, first step
        ACj[:-1, :-1] = Cj_log_distance_matrix
        # Fill all y items, second step
        ACj[:-1, Kj] = 1.0
        # Fill all z items, third step
        ACj[Kj, :-1] = 1.0
        # Supplement: bottom right element should be zero
        ACj[Kj, Kj] = 0.0
        # Construct constant vector
        bCj[:-1] = -np.log(np.abs(z_jw - z_jwc))
        # Compute 2-norm condition number
        kappa2Cj = np.linalg.cond(ACj)
        print(f"  Coefficient matrix 2-norm condition number for C{j} = {kappa2Cj:.4e}")
        # Linear system solution
        try:
            XCj = np.linalg.solve(ACj,bCj)
        except np.linalg.LinAlgError as err:
            print("Ill-conditioned matrix, solution failure!")
        # Solution separation, Q_{k} and Gamma
        QCj = XCj[0:Kj]
        Gamma_j = XCj[Kj]
        #print(f"  Gamma_{j} = ", Gamma_j)
    return QCj, Gamma_j

def backward_mapping_solution(z_jw, zeta_j, x_j, j, N):
    # Construct coefficient matrix and constant vector
    # K_{j}, Quantity of collocation points and charge points
    Kj = zeta_j.size
    # Construct coefficient matrix and constant vector
    Acj = np.zeros((Kj, Kj), dtype = np.complex128)
    bcj = np.zeros(Kj, dtype = np.complex128)
    # Construct coefficient matrix
    # Preparation: array dimension extension of collocation points and charge points
    # Prepare coefficient matrix in two steps
    # Step 1: prepare all x items
    # | x x x ... x x|
    # | x x x ... x x|
    # | x x x ... x x|
    # | . . . ... . .|
    # | x x x ... x x|
    # | x x x ... x x|
    zeta_jl_matrix = zeta_j[:, np.newaxis]
    x_jk_matrix = x_j[np.newaxis, :]
    # broadcast technique
    denominator = zeta_jl_matrix - x_jk_matrix
    Acj = 1. / denominator 
    if j == N:
				# Step 2: prepare all y items
       	# | x x x ... x y|
        # | x x x ... x y|
        # | x x x ... x y|
        # | . . . ... . .|
        # | x x x ... x y|
        # | x x x ... x y|
        Acj[:,0] = 1.
        # print("  Exterior periphery, no replacement necessary!")
    else:
				# Step 2: prepare all y items
        # | x x x ... x y|
        # | x x x ... x y|
        # | x x x ... x y|
        # | . . . ... . .|
        # | x x x ... x y|
        # | x x x ... x y|
        Acj[:,0] = 1.
        # Step 3 (optional for interior peripheries): replace last column with y items
        # | x x x ... z y|
        # | x x x ... z y|
        # | x x x ... z y|
        # | . . . ... . .|
        # | x x x ... z y|
        # | x x x ... z y|
        Acj[:,1] = zeta_j
    bcj = z_jw
        # Compute 2-norm condition number
    kappa2cj = np.linalg.cond(Acj)
    print(f"  Coefficient matrix 2-norm condition number for c{j} = {kappa2cj:.4e}")
        # Linear system solution
    try:
        xcj = np.linalg.solve(Acj,bcj)
    except np.linalg.LinAlgError as err:
        print("Ill-conditioned matrix, solution failure!")
    qcj = xcj
    return qcj

## Forward conformal mapping of Charge Simulation Method
def forward_mapping(z_jw, X_jw, Gamma_j, Q_j, j, z_jwc, N):
    # z_jw : sample points
        # Z_jw : charge points
        # Gamma_j : capacity constant
        # Q_j : charge
        # M_{j} sample points along cavity periphery for mapping
        M_j = z_jw.size
        # K_{j} charge points along cavity periphery
        K_j = X_jw.size
        # broadcast technique
        z_jwb = z_jw[:, np.newaxis]
        X_jwb = X_jw[np.newaxis, :]
        # |z_{j}^{(w)} - Z_{j}^{(w)}|
        abs_z = np.abs(z_jwb - X_jwb)
        # ln |z_{j}^{(w)} - Z_{j}^{(w)}|
        ln_z = np.log(abs_z)
        # 1 - \frac{Z_{j}^{(w)}}{z_{j}^{(w)}}
        if j == N:
            frac = 1.0 - z_jwb/X_jwb
        else:
            frac = 1.0 - X_jwb/z_jwb
        # \arg(1 - \frac{Z_{j}^{(w)}}{z_{j}^{(w)}})
        arg_z = np.angle(frac)
        # Zeta_mat = ln |z_{j}^{(w)} - Z_{j}^{(w)}| + i * \arg(1 - \frac{Z_{j}^{(w)}}{z_{j}^{(w)}}), a two-dimensional matrix
        Zeta_mat = ln_z + 1j * arg_z
        # \sum Q_{j,k} * Zeta_mat, matrix muplication
        sum_total = Zeta_mat @ Q_j
        # exponent
        exponent = Gamma_j + sum_total
        # zeta
        zeta_j = (z_jw - z_jwc) * np.exp(exponent)
        return zeta_j

def backward_mapping(zeta_grid, x_j, qcj, j, N):	
    zeta_col = np.asarray(zeta_grid)[:, np.newaxis]
    if j == N:	
        x_j_trunc = x_j[1:]	
        qcj_trunc = qcj[1:]
        x_row_N = x_j_trunc[np.newaxis, :]
        denominator_N = zeta_col - x_row_N
        denominator_N += 1e-12*(1 + 1j)
        term_sum = np.sum(qcj_trunc / denominator_N, axis = -1)
        term_cons = qcj[0]
        z_result = term_sum + term_cons
    else:
        x_j_trunc = x_j[2:]	
        qcj_trunc = qcj[2:]
        x_row = x_j_trunc[np.newaxis, :]
        denominator = zeta_col - x_row
        denominator += 1e-12*(1 + 1j)
        term_sum = np.sum(qcj_trunc / denominator, axis = -1)
        term_linear = qcj[1] * zeta_grid
        term_cons = qcj[0]
        z_result = term_sum + term_linear + term_cons
    return z_result

def backward_mapping_d(zeta, x_j, qcj, j, N):
    zeta_col = np.asarray(zeta)[:, np.newaxis]
    if j == N:	
        x_j_trunc = x_j[1:]	
        qcj_trunc = qcj[1:]
        x_row_N = x_j_trunc[np.newaxis, :]
        denominator_N = zeta_col - x_row_N
        denominator_N += 1e-12*(1 + 1j)
        z_result = np.sum(- qcj_trunc / denominator_N**2, axis = -1)
    else:
        x_j_trunc = x_j[2:]	
        qcj_trunc = qcj[2:]
        x_row = x_j_trunc[np.newaxis, :]
        denominator = zeta_col - x_row
        denominator += 1e-12*(1 + 1j)
        term_sum = np.sum( - qcj_trunc / denominator**2, axis = -1)
        term_linear = qcj[1]
        z_result = term_sum + term_linear
    return z_result

def backward_mapping_d2(zeta, x_j, qcj, j, N):
    zeta_col = np.asarray(zeta)[:, np.newaxis]
    if j == N:	
        x_j_trunc = x_j[1:]	
        qcj_trunc = qcj[1:]
        x_row_N = x_j_trunc[np.newaxis, :]
        denominator_N = zeta_col - x_row_N
        denominator_N += 1e-12*(1 + 1j)
        z_result = np.sum(2*qcj_trunc / denominator_N**3, axis = -1)
    else:
        x_j_trunc = x_j[2:]	
        qcj_trunc = qcj[2:]
        x_row = x_j_trunc[np.newaxis, :]
        denominator = zeta_col - x_row
        denominator += 1e-12*(1 + 1j)
        z_result = np.sum(2*qcj_trunc / denominator**3, axis = -1)
    return z_result

## Define a point is within periphery C1 or not
def inside_C1(z, r):
        x = np.real(z)
        y = np.imag(z)
        distance = x**2 + y**2
        return distance <= r**2 + 1e-12

## Define a point is within periphery C2 or not
def inside_C2(z, r, x0, x1, y0, y1):
        L = x0 + r
        R = x1 - r
        B = y0 + r
        T = y1 - r
        x = np.real(z)
        y = np.imag(z)
        xc = np.clip(x, L, R)
        yc = np.clip(y, B, T)
        distance = np.sqrt((x - xc)**2 + (y - yc)**2)
        return distance <= r + 1e-12

## Define a point is within periphery C3 or not
def inside_C3(z, a, b):
        x = np.real(z)
        y = np.imag(z)
        distance = (x / a)**2 + (y / b)**2
        return distance <= 1. + 1e-12

## Define a point is within periphery C4 or not
def inside_C4(z, a, bup, blw):
        x = np.real(z)
        y = np.imag(z)
        up_circle = (y >= 0.) & ((x / a)**2 + (y / bup)**2 <= 1. + 1e-12)
        lw_ellipse = (y < 0.) & ((x / a)**2 + (y / blw)**2 <= 1. + 1e-12)
        inside = up_circle | lw_ellipse
        return inside

## Define a point is within periphery C5 or not
def inside_C0(z, r, x0, x1, y0, y1):
        L = x0 + r
        R = x1 - r
        B = y0 + r
        T = y1 - r
        x = np.real(z)
        y = np.imag(z)
        xc = np.clip(x, L, R)
        yc = np.clip(y, B, T)
        distance = np.sqrt((x - xc)**2 + (y - yc)**2)
        return distance <= r + 1e-12
