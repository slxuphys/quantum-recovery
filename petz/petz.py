import numpy as np
import sympy as sp
from qsvt_sign import qsvt_pq
def QSVT(x, thetas):
    k = len(thetas)
    w = np.array([[x, -np.sqrt(1-x**2)],[np.sqrt(1-x**2), x]])
    w = np.moveaxis(w,2,0)
    u = np.eye(2)
    for i in range(0,k):
        u = np.diag([np.exp(1j*thetas[i]), -1.0]) @ w @ u
    u = w @ u
    v = np.array([1,0])
    p = v@u@v
    p 
    return p

def QSVT_sym(thetas):
    x = sp.symbols('x')
    k = len(thetas)
    w = np.array([[x, -sp.sqrt(1-x**2)],[sp.sqrt(1-x**2), x]])
    u = np.eye(2)
    for i in range(0,k):
        u = np.diag([np.exp(1j*thetas[i]), -1.0]) @ w @ u
    u = w @ u
    v = np.array([1,0])
    return v@u@v

def generate_random_encoding(U, D, A=1, sigma=None, entanglement=False):
    # Generate a random encoding matrix
    # U: encoding unitary
    # A: number of logical qubits
    # D: number of physical qubits
    # sigma: auxiliary parameter for the normal distribution
    if sigma is None:
        sigma = np.eye(2**A)/2**A # fully mixed state by default
    N = int(np.log(U.shape[0]) / np.log(2)) # number of qubits
    B = N-A; C = N-D
    U = np.reshape(U, (2**C, 2**D, 2**A, 2**B));
    if entanglement:
        E = np.transpose(U, (0, 1, 3, 2))
        E = np.reshape(E, (2**C, 2**(D+B), 2**A))
        E = E/2**(B/2)
        D = D + B
    else:
        E = U[:,:,:,0]
    E = np.transpose(E, (1, 0,2))
    sigma_sqrt = svt(sigma, lambda x: np.sqrt(x))
    E = np.einsum('ijk, kl->ijl', E, sigma_sqrt)
    E = np.reshape(E, (2**(D), 2**(C+A)))
    return E

def svt(M, f):
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    if callable(f):
        return (U * f(S)) @ Vh

    return [(U * func(S)) @ Vh for func in f]

def generate_random_unitary(N, seed=None):
    # Generate a Haar-random unitary matrix of size 2^N x 2^N
    rng = np.random.default_rng(seed)

    d = 2**N
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))

    Q, R = np.linalg.qr(X)

    diag_R = np.diagonal(R)
    ph = diag_R / np.abs(diag_R)

    Q = Q * ph.conj()
    return Q

def Approx_Petz_entanglement_fidelity(
        U, 
        D, A=1,
        angle=None,
        sigma=None, tau=None, 
        entanglement=False):
    """Return unconditional and postselected entanglement fidelities.

    angle=None gives exact Petz recovery; angle=[] gives zero QSVT iterations.
    """
    N = int(np.log(U.shape[0]) / np.log(2)) # number of qubits
    B = N-A; C = N-D
    if sigma is None:
        sigma = np.eye(2**A)/2**A # fully mixed state by default
    if tau is None:
        tau = np.eye(2**A)/2**A # fully mixed state by default
    sigma_invsqrt = svt(sigma, inv_sqrt)
    v = sigma_invsqrt @ tau
    v = v.ravel()
    E = generate_random_encoding(U, D, A, sigma, entanglement)
    E_d = E.conj().T
    if entanglement:
        alpha = np.sqrt(2**D/2**B)
        D = D + B # Hayden-Preskill setting
    else:
        alpha = np.sqrt(np.min([2**(A+C), 2**D]))
    # print('normalization alpha:', alpha)
    if angle is None:
        P=lambda x:np.ones_like(x)
        Q=lambda x:np.zeros_like(x)
    else:
        P = lambda x: qsvt_pq(x/alpha, angle)[0]
        Q = lambda x: qsvt_pq(x/alpha, angle)[1]
    E_dp, E_dq = svt(E_d, [P, lambda x: np.array(x/alpha)*Q(x)])
    # Keep axes in environment C, logical A, output D order.
    E_d = np.reshape(E_d, (2**C, 2**A, 2**D))
    E_dp = np.reshape(E_dp, (2**C, 2**A, 2**D))
    E_dq = np.reshape(E_dq, (2**C, 2**A, 2**D))
    # Output index pairs match vec(sigma^(-1/2) tau).
    contraction = 'eiz,cjz,ekw,clw->klij'
    path = np.einsum_path(contraction, E_d.conj(), E_dp, E_d, E_dp.conj(),
                          optimize='optimal')[0]
    Mp = np.einsum(contraction, E_d.conj(), E_dp, E_d, E_dp.conj(),
                   optimize=path)
    Mq = np.einsum(contraction, E_d.conj(), E_dq, E_d, E_dq.conj(),
                   optimize=path)
    Mp = Mp.reshape(2**(2*A), 2**(2*A))
    Mq = Mq.reshape(2**(2*A), 2**(2*A))
    # Partial trace over C and D in the unprojected failure contribution.
    logical_q = np.einsum('cad,cbd->ab', E_dq, E_dq.conj(), optimize=True)
    Mi = alpha**2 * np.kron(logical_q, sigma.T)
    rho = Mp - Mq + Mi

    # Success probability for the actual input tau; equals trace(Mp) if tau=sigma.
    H = sigma_invsqrt @ tau @ sigma_invsqrt
    p_succ = np.einsum('ijkj,ki->', Mp.reshape(2**A, 2**A, 2**A, 2**A), H)
    p_succ = np.real_if_close(p_succ)
    fidelity = np.real_if_close(v.conj() @ rho @ v)
    success_overlap = np.real_if_close(v.conj() @ Mp @ v)
    conditional = success_overlap / p_succ if p_succ > 0 else np.nan
    return fidelity, conditional



def inv_sqrt(x, eps=1e-10):
    out = np.zeros_like(x, dtype=float)
    mask = x > eps
    out[mask] = 1 / np.sqrt(x[mask])
    return out

