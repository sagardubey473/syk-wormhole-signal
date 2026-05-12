"""Fix 3: Timing test for N=14 transmission feasibility.

Tests each step individually to identify the bottleneck:
1. Majorana operator construction (28 ops on 14 qubits)
2. Hamiltonian construction (1001 coupling terms per side)
3. TFD construction (uses dim=128 single-side diag)
4. Full eigendecomposition of coupled H (16384 x 16384)

Reports whether full computation or Krylov alternative is needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from scipy import sparse

print("Fix 3: N=14 Timing Test")
print("=" * 60)
print(f"  N_per_side = 14, dim = {2**14} = 16384")
print(f"  C(14,4) = 1001 coupling terms per side")
print()

# Step 1: Build Majoranas
print("Step 1: Building 28 Majorana operators on 14 qubits (sparse)...")
t0 = time.time()
from src.majorana import build_majorana_operators
majoranas = build_majorana_operators(28, use_sparse=True)
t1 = time.time()
print(f"  Done in {t1-t0:.2f}s")
print(f"  Each operator: {majoranas[0].shape}, nnz={majoranas[0].nnz}")
print()

# Step 2: Build DoubledSYK Hamiltonian
print("Step 2: Building DoubledSYK(N=14, seed=0, p=1.0)...")
t2 = time.time()
from src.doubled import DoubledSYK
doubled = DoubledSYK(N_per_side=14, seed=0, sparsity=1.0, use_sparse=True)
t3 = time.time()
print(f"  Done in {t3-t2:.2f}s")
if sparse.issparse(doubled.H_L):
    print(f"  H_L: nnz={doubled.H_L.nnz}, density={doubled.H_L.nnz/16384**2:.4f}")
    print(f"  H_0: nnz={doubled.H_0.nnz}, density={doubled.H_0.nnz/16384**2:.4f}")
print()

# Step 3: Build coupled H and TFD
print("Step 3: Building coupled H (mu=0.1) and TFD (beta=8)...")
t4 = time.time()
H_coupled = doubled.build_coupled_hamiltonian(0.1)
from src.tfd import build_tfd
tfd, Z = build_tfd(doubled, 8.0)
t5 = time.time()
print(f"  Done in {t5-t4:.2f}s")
if sparse.issparse(H_coupled):
    print(f"  H_coupled: nnz={H_coupled.nnz}, density={H_coupled.nnz/16384**2:.4f}")
print(f"  TFD norm: {np.linalg.norm(tfd):.6f}")
print()

# Step 4: Test eigendecomposition (THIS IS THE BIG ONE)
print("Step 4: Full eigendecomposition of H_coupled (16384 x 16384)...")
print("  Converting to dense...")
t6 = time.time()
if sparse.issparse(H_coupled):
    H_dense = H_coupled.toarray()
else:
    H_dense = H_coupled
H_dense = 0.5 * (H_dense + H_dense.conj().T)  # enforce symmetry
t7 = time.time()
print(f"  Dense conversion: {t7-t6:.2f}s")
print(f"  Matrix memory: {H_dense.nbytes / 1e9:.2f} GB")
print(f"  Starting eigh... (this may take 5-30 minutes)")
sys.stdout.flush()

t8 = time.time()
evals, evecs = np.linalg.eigh(H_dense)
t9 = time.time()
print(f"  Eigendecomposition: {t9-t8:.1f}s ({(t9-t8)/60:.1f} min)")
print(f"  Spectrum: [{evals[0]:.4f}, {evals[-1]:.4f}], bandwidth={evals[-1]-evals[0]:.4f}")
print()

# Step 5: Transmission signal computation estimate
print("Step 5: Transmission signal timing (1 time point)...")
t10 = time.time()
tfd_eig = evecs.conj().T @ tfd
dE = np.subtract.outer(evals, evals)

N_sites = 14
psi_L = doubled.psi_L
psi_R = doubled.psi_R

# Compute for one site
pL = psi_L[0].toarray() if sparse.issparse(psi_L[0]) else psi_L[0]
pR = psi_R[0].toarray() if sparse.issparse(psi_R[0]) else psi_R[0]
ket = pL @ tfd
ket_eig = evecs.conj().T @ ket
R_eig = evecs.conj().T @ pR @ evecs
A = np.outer(tfd_eig.conj(), ket_eig) * R_eig

phases = np.exp(1j * dE * 5.0)  # one time point
C_one = np.sum(A * phases)
t11 = time.time()
print(f"  One site, one time: {t11-t10:.2f}s")
print(f"  Signal value: |C| = {abs(C_one):.6f}")
print()

# Summary
total = t9 - t2  # construction + diag
print("=" * 60)
print("FEASIBILITY SUMMARY")
print("=" * 60)
print(f"  Construction (DoubledSYK + coupled H + TFD): {t5-t2:.1f}s")
print(f"  Eigendecomposition: {t9-t8:.1f}s")
print(f"  Signal per site per time: {(t11-t10)/1:.2f}s")
print(f"  Total per realization: ~{t5-t2 + (t9-t8) + 14*120*(t11-t10):.0f}s")
print()

time_per_real = t5 - t2 + (t9 - t8) + 14 * 0.1  # rough
if time_per_real < 300:
    print(f"  FEASIBLE: ~{time_per_real:.0f}s/realization")
    print(f"  50 real x 5 sparsity = {250*time_per_real/3600:.1f} hours")
    print(f"  30 real x 3 sparsity = {90*time_per_real/3600:.1f} hours")
elif time_per_real < 600:
    print(f"  MARGINAL: ~{time_per_real:.0f}s/realization")
    print(f"  Recommend: 30 real x 3 sparsity = {90*time_per_real/3600:.1f} hours")
else:
    print(f"  INFEASIBLE at full scale: ~{time_per_real:.0f}s/realization")
    print(f"  Need Krylov approach or reduced scope")
