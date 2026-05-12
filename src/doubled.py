"""Doubled SYK system: left + right copies sharing disorder.

Constructs 2*N_per_side Majorana operators on a single Jordan-Wigner chain
of N_per_side qubits. Left Majoranas use qubits 0..N_per_side/2-1,
Right Majoranas use qubits N_per_side/2..N_per_side-1.

See CONVENTIONS.md section 5.

References:
    - Maldacena & Qi, arXiv:1804.00491, eq. (2.1)
"""

import numpy as np
from scipy import sparse
from itertools import combinations
from .majorana import build_majorana_operators, verify_anticommutation


class DoubledSYK:
    """Doubled SYK system with left and right copies.

    Parameters
    ----------
    N_per_side : int
        Number of Majorana fermions per side (must be even).
    seed : int
        Random seed for disorder (shared between L and R).
    J : float
        Energy scale.
    sparsity : float
        Fraction of couplings retained, in (0, 1].
    use_sparse : bool or None
        Auto-select if None.
    """

    def __init__(self, N_per_side, seed, J=1.0, sparsity=1.0, use_sparse=None):
        if N_per_side < 4 or N_per_side % 2 != 0:
            raise ValueError(f"N_per_side must be even and >= 4, got {N_per_side}")

        self.N_per_side = N_per_side
        self.N_total = 2 * N_per_side
        self.seed = seed
        self.J = J
        self.sparsity = sparsity
        self.n_qubits = N_per_side  # total qubits for both sides
        self.dim = 2 ** self.n_qubits

        if use_sparse is None:
            use_sparse = self.n_qubits > 6
        self.use_sparse = use_sparse

        # Build all 2*N_per_side Majorana operators on the full chain
        self.majoranas = build_majorana_operators(self.N_total, use_sparse=use_sparse)

        # Split into left and right
        self.psi_L = self.majoranas[:N_per_side]      # indices 0..N_per_side-1
        self.psi_R = self.majoranas[N_per_side:]       # indices N_per_side..2*N_per_side-1

        # Generate shared disorder couplings
        self.couplings = self._generate_couplings()

        # Build left and right Hamiltonians
        self.H_L = self._build_side_hamiltonian(self.psi_L)
        self.H_R = self._build_side_hamiltonian(self.psi_R)

        # Uncoupled Hamiltonian
        self.H_0 = self.H_L + self.H_R

        # Verify Hermiticity
        self._check_hermiticity()

    def _generate_couplings(self):
        """Generate coupling tensor J_{ijkl} with correct variance.

        Same as SYKHamiltonian but uses N_per_side for the coupling count.
        """
        rng = np.random.RandomState(self.seed)
        N = self.N_per_side
        sigma = np.sqrt(6.0 * self.J**2 / (N**3))

        couplings = {}
        indices = list(combinations(range(N), 4))

        if self.sparsity < 1.0:
            sigma_sparse = sigma / np.sqrt(self.sparsity)
            all_J = rng.normal(0, sigma_sparse, size=len(indices))
            mask = rng.random(len(indices)) < self.sparsity
            for idx, (ijkl, j_val) in enumerate(zip(indices, all_J)):
                if mask[idx]:
                    couplings[ijkl] = j_val
        else:
            all_J = rng.normal(0, sigma, size=len(indices))
            for ijkl, j_val in zip(indices, all_J):
                couplings[ijkl] = j_val

        return couplings

    def _build_side_hamiltonian(self, psi_ops):
        """Build SYK Hamiltonian for one side.

        H = -(1/4!) sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l

        Parameters
        ----------
        psi_ops : list
            Majorana operators for this side (in the doubled Hilbert space).
        """
        prefactor = -1.0 / 24.0
        N = self.N_per_side

        if self.use_sparse:
            H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)

        for (i, j, k, l), J_val in self.couplings.items():
            prod = psi_ops[i] @ psi_ops[j] @ psi_ops[k] @ psi_ops[l]
            H = H + (prefactor * J_val) * prod

        if self.use_sparse:
            H = H.tocsr()

        return H

    def _check_hermiticity(self, tol=1e-10):
        """Verify all Hamiltonians are Hermitian."""
        for name, H in [('H_L', self.H_L), ('H_R', self.H_R), ('H_0', self.H_0)]:
            if self.use_sparse:
                diff = H - H.conj().T
                err = sparse.linalg.norm(diff)
            else:
                diff = H - H.conj().T
                err = np.linalg.norm(diff)
            if err > tol:
                raise RuntimeError(f"{name} not Hermitian! Error = {err:.2e}")

    def build_interaction(self, mu):
        """Build interaction Hamiltonian H_int = i * mu * sum_i psi^L_i psi^R_i.

        This is Hermitian because psi^L and psi^R anticommute (different
        qubit support), so (i * psi^L psi^R)^dagger = -i * psi^R psi^L
        = +i * psi^L psi^R.

        Ref: MQ eq. (2.1).

        Parameters
        ----------
        mu : float
            Coupling strength.

        Returns
        -------
        H_int : ndarray or sparse matrix
        """
        if self.use_sparse:
            H_int = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H_int = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.N_per_side):
            H_int = H_int + 1j * mu * (self.psi_L[i] @ self.psi_R[i])

        if self.use_sparse:
            H_int = H_int.tocsr()

        # Verify Hermiticity
        if self.use_sparse:
            err = sparse.linalg.norm(H_int - H_int.conj().T)
        else:
            err = np.linalg.norm(H_int - H_int.conj().T)
        if err > 1e-10:
            raise RuntimeError(f"H_int not Hermitian! Error = {err:.2e}")

        return H_int

    def build_coupled_hamiltonian(self, mu):
        """Build full coupled Hamiltonian H = H_L + H_R + H_int.

        Parameters
        ----------
        mu : float
            Coupling strength.

        Returns
        -------
        H : ndarray or sparse matrix
        """
        H_int = self.build_interaction(mu)
        H = self.H_0 + H_int

        # Verify Hermiticity
        if self.use_sparse:
            err = sparse.linalg.norm(H - H.conj().T)
        else:
            err = np.linalg.norm(H - H.conj().T)
        if err > 1e-10:
            raise RuntimeError(f"Coupled H not Hermitian! Error = {err:.2e}")

        return H

    def get_H_L_dense(self):
        """Return dense H_L."""
        if self.use_sparse:
            return self.H_L.toarray()
        return self.H_L.copy()

    def get_H_R_dense(self):
        """Return dense H_R."""
        if self.use_sparse:
            return self.H_R.toarray()
        return self.H_R.copy()

    def get_H_0_dense(self):
        """Return dense H_0."""
        if self.use_sparse:
            return self.H_0.toarray()
        return self.H_0.copy()
