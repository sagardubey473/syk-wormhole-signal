"""SYK Hamiltonian construction and analysis.

Builds the q=4 SYK Hamiltonian:
    H = -(1/4!) sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l

with coupling variance <J^2> = 6 J^2 / N^3 (J=1).
See CONVENTIONS.md section 3.

References:
    - Maldacena & Stanford, arXiv:1604.07818, eq. (2.2)-(2.3)
    - Polchinski & Rosenhaus, arXiv:1601.06768
"""

import numpy as np
from scipy import sparse
from itertools import combinations
from .majorana import build_majorana_operators


class SYKHamiltonian:
    """SYK q=4 Hamiltonian with optional sparsification.

    Parameters
    ----------
    N : int
        Number of Majorana fermions (must be even).
    seed : int
        Random seed for coupling disorder.
    J : float
        Energy scale (default 1).
    sparsity : float
        Fraction of couplings retained, in (0, 1]. p=1 is dense SYK.
        Surviving couplings have variance scaled by 1/p.
    use_sparse : bool or None
        Use sparse matrix representation. Auto-selects if None.
    """

    def __init__(self, N, seed, J=1.0, sparsity=1.0, use_sparse=None):
        if N < 4 or N % 2 != 0:
            raise ValueError(f"N must be even and >= 4, got {N}")
        if not (0 < sparsity <= 1):
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")

        self.N = N
        self.seed = seed
        self.J = J
        self.sparsity = sparsity
        self.n_qubits = N // 2
        self.dim = 2 ** self.n_qubits

        if use_sparse is None:
            use_sparse = self.n_qubits > 6
        self.use_sparse = use_sparse

        # Build Majorana operators
        self.majoranas = build_majorana_operators(N, use_sparse=use_sparse)

        # Generate couplings and build Hamiltonian
        self.couplings = self._generate_couplings()
        self.hamiltonian = self._build_hamiltonian()

        # Verify Hermiticity
        self._check_hermiticity()

        # Cache for diagonalization
        self._eigenvalues = None
        self._eigenvectors = None

    def _generate_couplings(self):
        """Generate J_{ijkl} couplings with correct variance.

        Variance: <J^2> = 6 J^2 / (N^3 * p) where p is sparsity.
        The 1/p factor compensates for the fraction of zeroed couplings.
        """
        rng = np.random.RandomState(self.seed)
        N = self.N
        # Variance per coupling: 6 * J^2 / N^3
        # With sparsity: variance scaled by 1/p for surviving couplings
        sigma = np.sqrt(6.0 * self.J**2 / (N**3))

        couplings = {}
        indices = list(combinations(range(N), 4))

        if self.sparsity < 1.0:
            # Sparse SYK: retain each coupling with probability p
            sigma_sparse = sigma / np.sqrt(self.sparsity)
            # Draw all couplings first for seed consistency
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

    def _build_hamiltonian(self):
        """Build H = -(1/4!) sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l.

        The prefactor is i^2/4! = -1/24.
        Ref: MS eq. (2.2). The i^2 comes from writing H in terms of
        Majorana bilinears; since each psi is Hermitian and J is real,
        the -1 ensures H is Hermitian.
        """
        prefactor = -1.0 / 24.0  # i^2 / 4! = -1/24

        if self.use_sparse:
            H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)

        psi = self.majoranas
        for (i, j, k, l), J_val in self.couplings.items():
            # psi_i psi_j psi_k psi_l
            prod = psi[i] @ psi[j] @ psi[k] @ psi[l]
            H = H + (prefactor * J_val) * prod

        if self.use_sparse:
            H = H.tocsr()

        return H

    def _check_hermiticity(self, tol=1e-10):
        """Verify Hamiltonian is Hermitian."""
        if self.use_sparse:
            diff = self.hamiltonian - self.hamiltonian.conj().T
            err = sparse.linalg.norm(diff)
        else:
            diff = self.hamiltonian - self.hamiltonian.conj().T
            err = np.linalg.norm(diff)

        if err > tol:
            raise RuntimeError(
                f"SYK Hamiltonian not Hermitian! Error = {err:.2e}"
            )
        self.hermiticity_error = err

    def diagonalize(self):
        """Compute eigenvalues and eigenvectors.

        Returns
        -------
        eigenvalues : ndarray, shape (dim,)
        eigenvectors : ndarray, shape (dim, dim)
            Column i is the eigenvector for eigenvalue i.
        """
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors

        if self.use_sparse:
            H_dense = self.hamiltonian.toarray()
        else:
            H_dense = self.hamiltonian

        # Force Hermitian solve for numerical stability
        H_herm = 0.5 * (H_dense + H_dense.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(H_herm)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        return eigenvalues, eigenvectors

    def spectrum_statistics(self):
        """Compute level spacing ratio statistics.

        Returns
        -------
        r_values : ndarray
            Individual spacing ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).
        r_mean : float
            Mean spacing ratio <r>.
        """
        evals, _ = self.diagonalize()
        spacings = np.diff(evals)

        # Remove near-zero spacings (exact degeneracies)
        spacings = spacings[spacings > 1e-14]

        r_values = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
        r_mean = np.mean(r_values)
        return r_values, r_mean

    def thermal_density_matrix(self, beta):
        """Compute rho = exp(-beta * H) / Z.

        Returns
        -------
        rho : ndarray, shape (dim, dim)
        Z : float
            Partition function.
        """
        evals, evecs = self.diagonalize()
        boltzmann = np.exp(-beta * evals)
        Z = np.sum(boltzmann)
        # rho = V diag(exp(-beta*E)/Z) V^dagger
        rho = evecs @ np.diag(boltzmann / Z) @ evecs.conj().T
        return rho, Z

    def two_point_function(self, beta, t_array, site=0):
        """Compute thermal two-point function G(t) = <psi_i(t) psi_i(0)>_beta.

        G(t) = Tr[rho * psi_i(t) * psi_i(0)]
             = Tr[rho * e^{iHt} psi_i e^{-iHt} * psi_i]

        Parameters
        ----------
        beta : float
            Inverse temperature.
        t_array : ndarray
            Time points.
        site : int
            Majorana index.

        Returns
        -------
        G : ndarray, shape (len(t_array),)
            Complex-valued two-point function.
        """
        evals, evecs = self.diagonalize()
        boltzmann = np.exp(-beta * evals)
        Z = np.sum(boltzmann)

        # psi in eigenbasis
        psi = self.majoranas[site]
        if sparse.issparse(psi):
            psi = psi.toarray()
        psi_eig = evecs.conj().T @ psi @ evecs  # dim x dim

        G = np.zeros(len(t_array), dtype=complex)
        for idx, t in enumerate(t_array):
            # e^{iE_m t} (psi)_{mn} e^{-iE_n t} (psi)_{nm} * rho_mm
            phases_L = np.exp(1j * evals * t)    # e^{iE_m t}
            phases_R = np.exp(-1j * evals * t)   # e^{-iE_n t}

            # G(t) = sum_{m,n} (rho_m) * e^{i(E_m-E_n)t} * |psi_{mn}|^2
            # where rho_m = e^{-beta E_m}/Z
            for m in range(self.dim):
                G[idx] += (boltzmann[m] / Z) * np.sum(
                    psi_eig[m, :] * psi_eig[:, m] *
                    np.exp(1j * (evals[m] - evals) * t)
                )

        return G

    def otoc(self, beta, t_array, site_i=0, site_j=1):
        """Compute OTOC F(t) = Tr[rho^{1/4} W(t) rho^{1/4} V rho^{1/4} W(t) rho^{1/4} V].

        Uses regulated form with rho^{1/4} insertions for finite temperature.
        W = psi_i, V = psi_j.

        Ref: MSS arXiv:1503.01409, eq. (1.2).

        Parameters
        ----------
        beta : float
        t_array : ndarray
        site_i, site_j : int

        Returns
        -------
        F : ndarray, shape (len(t_array),)
        """
        evals, evecs = self.diagonalize()

        # rho^{1/4} in eigenbasis
        rho_quarter_eig = np.exp(-beta * evals / 4.0)
        rho_quarter_eig /= np.sum(np.exp(-beta * evals)) ** 0.25  # normalize

        # Operators in eigenbasis
        W = self.majoranas[site_i]
        V = self.majoranas[site_j]
        if sparse.issparse(W):
            W = W.toarray()
        if sparse.issparse(V):
            V = V.toarray()
        W_eig = evecs.conj().T @ W @ evecs
        V_eig = evecs.conj().T @ V @ evecs

        # rho^{1/4} as diagonal matrix in eigenbasis
        rho_q = np.diag(rho_quarter_eig)

        F = np.zeros(len(t_array), dtype=complex)
        for idx, t in enumerate(t_array):
            # W(t) in eigenbasis: e^{iEt} W e^{-iEt}
            phases = np.exp(1j * evals * t)
            Wt_eig = np.diag(phases) @ W_eig @ np.diag(phases.conj())

            # F = Tr[rho_q Wt rho_q V rho_q Wt rho_q V]
            A = rho_q @ Wt_eig @ rho_q @ V_eig @ rho_q @ Wt_eig @ rho_q @ V_eig
            F[idx] = np.trace(A)

        return F

    def get_hamiltonian_dense(self):
        """Return dense Hamiltonian matrix."""
        if self.use_sparse:
            return self.hamiltonian.toarray()
        return self.hamiltonian.copy()

    def coupling_variance(self):
        """Compute empirical variance of couplings."""
        if len(self.couplings) == 0:
            return 0.0
        vals = np.array(list(self.couplings.values()))
        return np.var(vals)

    def expected_coupling_variance(self):
        """Theoretical coupling variance: 6 J^2 / (N^3 * p)."""
        return 6.0 * self.J**2 / (self.N**3 * self.sparsity)
