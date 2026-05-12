# Conventions for Coupled SYK Traversable Wormhole Simulator

All sign, normalization, basis, and ordering conventions used throughout the code.
References: Maldacena-Stanford (MS, 1604.07818), Maldacena-Qi (MQ, 1804.00491),
Maldacena-Shenker-Stanford (MSS, 1503.01409).

## 1. Majorana Fermion Algebra

Anticommutation relation:
    {psi_i, psi_j} = 2 * delta_{ij}

This implies psi_i^2 = I (the identity matrix).
Each psi_i is Hermitian: psi_i^dagger = psi_i.

Reference: MS eq. (2.1).

## 2. Jordan-Wigner Construction

For N Majorana fermions, we use N/2 qubits. The mapping is:

    psi_{2k-1} = (prod_{j=1}^{k-1} Z_j) X_k      for k = 1, ..., N/2
    psi_{2k}   = (prod_{j=1}^{k-1} Z_j) Y_k       for k = 1, ..., N/2

where X_k, Y_k, Z_k are Pauli matrices acting on qubit k, and the product
is a Jordan-Wigner string ensuring anticommutation.

Qubit ordering: qubit 1 is the leftmost tensor factor.

## 3. SYK Hamiltonian

    H_SYK = (i^2 / 4!) sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l
          = -(1/4!) sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l

The factor i^2 = -1 ensures H is Hermitian when J_{ijkl} are real and
the psi_i are Hermitian.

Coupling distribution: J_{ijkl} drawn from N(0, sigma^2) where
    sigma^2 = 6 * J^2 / N^3

with J = 1 setting the energy scale. This gives the correct normalization
for the q=4 SYK model per MS eq. (2.2)-(2.3).

Reference: MS eq. (2.2), (2.3).

## 4. Units

- Energy: measured in units of J (= 1).
- Temperature: beta in units of 1/J.
- Time: t in units of 1/J.
- Coupling: mu in units of J.

## 5. Doubled System (Left + Right)

For N_per_side Majorana fermions per side, the doubled system has
2 * N_per_side Majorana fermions living on a single Jordan-Wigner chain
of N_per_side qubits total (N_per_side/2 qubits per side).

Ordering: Left Majoranas are indices 0, 1, ..., N_per_side-1.
          Right Majoranas are indices N_per_side, ..., 2*N_per_side-1.

The full Jordan-Wigner chain has N_per_side qubits:
- Qubits 0 to N_per_side/2 - 1 carry the left Majoranas.
- Qubits N_per_side/2 to N_per_side - 1 carry the right Majoranas.

H_L is built from left Majorana operators (indices 0..N_per_side-1)
using the SAME coupling tensor J_{ijkl}.
H_R is built from right Majorana operators (indices N_per_side..2*N_per_side-1)
using the SAME coupling tensor J_{ijkl} (shared disorder).

The uncoupled Hamiltonian: H_0 = H_L + H_R.
Verification: [H_L, H_R] = 0 since they act on different qubits.

## 6. Interaction Hamiltonian

    H_int = i * mu * sum_{i=0}^{N_per_side-1} psi^L_i psi^R_i

where psi^L_i is the i-th left Majorana and psi^R_i is the i-th right Majorana.

The factor of i ensures H_int is Hermitian:
    (i * psi^L psi^R)^dagger = -i * psi^R psi^L = +i * psi^L psi^R

since psi^L and psi^R anticommute (they are on different sites of the
Jordan-Wigner chain).

Reference: MQ eq. (2.1).

## 7. Full Coupled Hamiltonian

    H = H_L + H_R + H_int

## 8. Thermofield Double (TFD) State

    |TFD(beta)> = (1/sqrt(Z)) sum_n exp(-beta * E_n / 2) |n>_L |n*>_R

where:
- {|n>} are energy eigenstates of the single-side SYK Hamiltonian.
- E_n are the corresponding eigenvalues.
- Z = sum_n exp(-beta * E_n) is the partition function.
- |n*>_R denotes the "complex conjugate" state on the right side.

Conjugation convention:
In our Jordan-Wigner construction, H_L and H_R (restricted to their
respective tensor factors) are represented by the SAME matrix as H_single.
Therefore the right eigenstates |n>_R are the same vectors as |n>_L,
WITHOUT complex conjugation:

    |TFD(beta)> = (1/sqrt(Z)) sum_n exp(-beta * E_n / 2) |n>_L |n>_R

This differs from the abstract MQ convention |TFD> = sum e^{-beta E/2} |n>|n*>
where conjugation is needed because the right Hamiltonian is defined as
H_R = K H_L K^{-1} with K anti-unitary. In our explicit qubit representation,
the right Majoranas have the same matrix structure as the left on their
respective tensor factors.

In the doubled Hilbert space with basis |a>_L |b>_R, the TFD state vector
component is:
    <a|_L <b|_R |TFD> = (1/sqrt(Z)) sum_n exp(-beta*E_n/2) <a|n> <b|n>

Reference: MQ eq. (2.4).

## 9. Wormhole Transmission Signal

    C(t) = <TFD| psi^R_j(t) psi^L_j(0) |TFD>

where psi^R_j(t) = exp(iHt) psi^R_j exp(-iHt) is the Heisenberg operator
evolved under the FULL coupled Hamiltonian H.

The signal is averaged over Majorana site index j:
    C_avg(t) = (1/N_per_side) sum_{j=0}^{N_per_side-1} C_j(t)

We report |C_avg(t)| as the transmission signal magnitude.

Reference: MQ eq. (4.1)-(4.5), Figure 2.

## 10. Sparsification

Sparse SYK: retain each coupling J_{ijkl} independently with probability p.
Surviving couplings have variance scaled by 1/p to preserve total variance:
    <J_{ijkl}^2>_sparse = (6 * J^2) / (N^3 * p)

This ensures the spectral bandwidth is preserved on average.

## 11. Level Spacing Ratio

    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})

where s_n = E_{n+1} - E_n are consecutive level spacings (after sorting).

Predicted values (random matrix theory):
- GOE: <r> ~ 0.5307
- GUE: <r> ~ 0.6027
- Poisson: <r> ~ 0.3863

The SYK model at q=4 belongs to different random matrix ensembles depending
on N mod 8 (Bose-Fermi symmetry classification):
- N mod 8 = 0: GOE (time-reversal symmetric)
- N mod 8 = 2: GUE (no symmetry) ← use for chaos diagnostics
- N mod 8 = 4: BDI (antiunitary symmetry, reduced repulsion, <r> ~ 0.37)
- N mod 8 = 6: GUE (no symmetry) ← use for chaos diagnostics

RESTRICTION (established by project audit):
For chaos-to-Poisson transition studies (level spacing ratio, SFF diagnostics),
use ONLY N=10 (mod 8 = 2, GUE) and N=14 (mod 8 = 6, GUE). Do NOT use:
- N=8 (mod 8 = 0, GOE): too small (dim=16), finite-size artifacts dominate
- N=12 (mod 8 = 4, BDI): <r> ≈ 0.37 at ALL sparsities, indistinguishable
  from Poisson. Cannot be used to detect chaos-integrable transitions.

For other diagnostics (transmission peak, OTOC shape, mutual information),
all N values can be used since these do not depend on the RMT ensemble class.

## 12. OTOC (Out-of-Time-Order Correlator)

    F(t) = <psi_i(t) psi_j(0) psi_i(t) psi_j(0)>_beta

for i != j. The regulated thermal expectation value is:
    F(t) = Tr[rho^{1/4} psi_i(t) rho^{1/4} psi_j(0) rho^{1/4} psi_i(t) rho^{1/4} psi_j(0)]

where rho = exp(-beta H) / Z.

The Lyapunov exponent lambda_L is extracted from the early-time decay:
    1 - F(t)/F(0) ~ epsilon * exp(lambda_L * t)

Chaos bound (MSS): lambda_L <= 2*pi / beta.

Reference: MSS eq. (1.2).
