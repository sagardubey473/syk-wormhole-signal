"""Comparison (non-holographic) systems for RQ2 mimicry diagnostic.

Systems that may produce similar-looking transmission correlators
but lack holographic origin:

System A - Free fermion (random quadratic Hamiltonian)
System B - SYK_q with variable q (q=2 free, q=4 chaotic)
System C - Spin chain with tunable chaos
System D - Non-Gaussian SYK
"""

import numpy as np
from scipy import sparse
from itertools import combinations
from .majorana import build_majorana_operators


class FreeFermionSystem:
    """Random quadratic (free) fermion Hamiltonian.

    H = i * sum_{i<j} K_{ij} psi_i psi_j

    where K is antisymmetric. This is integrable (non-chaotic).
    """

    def __init__(self, N, seed, J=1.0, use_sparse=False):
        self.N = N
        self.seed = seed
        self.J = J
        self.n_qubits = N // 2
        self.dim = 2 ** self.n_qubits
        self.use_sparse = use_sparse

        self.majoranas = build_majorana_operators(N, use_sparse=use_sparse)
        self.hamiltonian = self._build_hamiltonian()

        # Verify Hermiticity
        H = self.hamiltonian
        if sparse.issparse(H):
            err = sparse.linalg.norm(H - H.conj().T)
        else:
            err = np.linalg.norm(H - H.conj().T)
        if err > 1e-10:
            raise RuntimeError(f"Free fermion H not Hermitian! Error = {err:.2e}")

        self._eigenvalues = None
        self._eigenvectors = None

    def _build_hamiltonian(self):
        rng = np.random.RandomState(self.seed)
        N = self.N
        # Random antisymmetric coupling: K_{ij} = -K_{ji}
        sigma = self.J * np.sqrt(2.0 / N)

        if self.use_sparse:
            H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(N):
            for j in range(i+1, N):
                K_ij = rng.normal(0, sigma)
                # H += i * K_{ij} * psi_i psi_j
                prod = self.majoranas[i] @ self.majoranas[j]
                H = H + 1j * K_ij * prod

        if self.use_sparse:
            H = H.tocsr()
        return H

    def diagonalize(self):
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors
        H = self.hamiltonian
        if sparse.issparse(H):
            H = H.toarray()
        H_herm = 0.5 * (H + H.conj().T)
        evals, evecs = np.linalg.eigh(H_herm)
        self._eigenvalues = evals
        self._eigenvectors = evecs
        return evals, evecs


class SYKq:
    """Generalized SYK_q model with variable body number q.

    q=2: free (integrable), q=4: standard SYK (maximally chaotic).
    """

    def __init__(self, N, q, seed, J=1.0, use_sparse=False):
        if q < 2 or q % 2 != 0:
            raise ValueError(f"q must be even and >= 2, got {q}")
        self.N = N
        self.q = q
        self.seed = seed
        self.J = J
        self.n_qubits = N // 2
        self.dim = 2 ** self.n_qubits
        self.use_sparse = use_sparse

        self.majoranas = build_majorana_operators(N, use_sparse=use_sparse)
        self.hamiltonian = self._build_hamiltonian()

        H = self.hamiltonian
        if sparse.issparse(H):
            err = sparse.linalg.norm(H - H.conj().T)
        else:
            err = np.linalg.norm(H - H.conj().T)
        if err > 1e-10:
            raise RuntimeError(f"SYK_q H not Hermitian! Error = {err:.2e}")

        self._eigenvalues = None
        self._eigenvectors = None

    def _build_hamiltonian(self):
        rng = np.random.RandomState(self.seed)
        N = self.N
        q = self.q
        # Variance: q! * J^2 / (q * N^{q-1}) -- generalized normalization
        from math import factorial
        sigma = np.sqrt(factorial(q) * self.J**2 / (q * N**(q-1)))
        prefactor = (1j)**(q // 2) / factorial(q)  # i^{q/2} / q!

        indices = list(combinations(range(N), q))

        if self.use_sparse:
            H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)

        all_J = rng.normal(0, sigma, size=len(indices))
        for idx_set, J_val in zip(indices, all_J):
            prod = self.majoranas[idx_set[0]]
            for k in range(1, q):
                prod = prod @ self.majoranas[idx_set[k]]
            H = H + prefactor * J_val * prod

        if self.use_sparse:
            H = H.tocsr()
        return H

    def diagonalize(self):
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors
        H = self.hamiltonian
        if sparse.issparse(H):
            H = H.toarray()
        H_herm = 0.5 * (H + H.conj().T)
        evals, evecs = np.linalg.eigh(H_herm)
        self._eigenvalues = evals
        self._eigenvectors = evecs
        return evals, evecs


class MixedFieldIsing:
    """Mixed-field Ising chain: tunable from integrable to chaotic.

    H = -J_z sum_i Z_i Z_{i+1} - h_x sum_i X_i - h_z sum_i Z_i

    Integrable when h_x=0 or h_z=0. Chaotic when both nonzero.
    """

    def __init__(self, n_qubits, J_z=1.0, h_x=0.5, h_z=0.5, seed=None,
                 use_sparse=False):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.J_z = J_z
        self.h_x = h_x
        self.h_z = h_z
        self.use_sparse = use_sparse

        self.hamiltonian = self._build_hamiltonian()
        self._eigenvalues = None
        self._eigenvectors = None

    def _build_hamiltonian(self):
        n = self.n_qubits
        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        H = np.zeros((self.dim, self.dim), dtype=complex)

        def pauli_at(pauli, site, n_sites):
            """Put Pauli operator at site, identity elsewhere."""
            op = np.eye(1)
            for s in range(n_sites):
                op = np.kron(op, pauli if s == site else I2)
            return op

        # ZZ interaction (open boundary conditions)
        for i in range(n - 1):
            ZZ = pauli_at(Z, i, n) @ pauli_at(Z, i+1, n)
            H -= self.J_z * ZZ

        # Transverse field
        for i in range(n):
            H -= self.h_x * pauli_at(X, i, n)
            H -= self.h_z * pauli_at(Z, i, n)

        return H

    def diagonalize(self):
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors
        H_herm = 0.5 * (self.hamiltonian + self.hamiltonian.conj().T)
        evals, evecs = np.linalg.eigh(H_herm)
        self._eigenvalues = evals
        self._eigenvectors = evecs
        return evals, evecs


class NonGaussianSYK:
    """SYK with non-Gaussian coupling distribution.

    Options: 'bimodal' (J = +/- J0) or 'uniform' (J ~ Uniform[-a, a]).
    May break the specific large-N structure underlying the gravity dual.
    """

    def __init__(self, N, seed, J=1.0, distribution='bimodal', use_sparse=False):
        self.N = N
        self.seed = seed
        self.J = J
        self.distribution = distribution
        self.n_qubits = N // 2
        self.dim = 2 ** self.n_qubits
        self.use_sparse = use_sparse

        self.majoranas = build_majorana_operators(N, use_sparse=use_sparse)
        self.hamiltonian = self._build_hamiltonian()

        H = self.hamiltonian
        if sparse.issparse(H):
            err = sparse.linalg.norm(H - H.conj().T)
        else:
            err = np.linalg.norm(H - H.conj().T)
        if err > 1e-10:
            raise RuntimeError(f"Non-Gaussian SYK H not Hermitian! Error = {err:.2e}")

        self._eigenvalues = None
        self._eigenvectors = None

    def _build_hamiltonian(self):
        rng = np.random.RandomState(self.seed)
        N = self.N
        sigma = np.sqrt(6.0 * self.J**2 / (N**3))
        indices = list(combinations(range(N), 4))
        prefactor = -1.0 / 24.0

        if self.use_sparse:
            H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)

        if self.distribution == 'bimodal':
            # J = +/- sigma with equal probability
            signs = rng.choice([-1, 1], size=len(indices))
            all_J = sigma * signs
        elif self.distribution == 'uniform':
            # Uniform with same variance: U[-a, a] with a = sigma*sqrt(3)
            a = sigma * np.sqrt(3)
            all_J = rng.uniform(-a, a, size=len(indices))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        psi = self.majoranas
        for (i, j, k, l), J_val in zip(indices, all_J):
            prod = psi[i] @ psi[j] @ psi[k] @ psi[l]
            H = H + prefactor * J_val * prod

        if self.use_sparse:
            H = H.tocsr()
        return H

    def diagonalize(self):
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors
        H = self.hamiltonian
        if sparse.issparse(H):
            H = H.toarray()
        H_herm = 0.5 * (H + H.conj().T)
        evals, evecs = np.linalg.eigh(H_herm)
        self._eigenvalues = evals
        self._eigenvectors = evecs
        return evals, evecs
