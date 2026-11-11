"""
appendix_a_check.py - numerical check of Appendix A (the pole-zero parameter M).

Pure NumPy; does not import or change any of the thesis code.
Run on the laptop:   python appendix_a_check.py

Part 1: exact M against the two approximations, at this study's operating point
        (50 MS/s, tau = 50 us) and at a short-tau case (tau/T = 5).
Part 2: apply "running sum + M x input" to a sampled exponential and check
        that it gives a flat step of height A(M+1) only for the exact M.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
"""
import numpy as np

def M_exact(T, tau):
    return 1.0 / np.expm1(T / tau)          # 1/(exp(T/tau) - 1), Eq. A.1

print("Part 1 - exact M vs approximations")
print(f"{'tau/T':>8} {'exact M':>16} {'tau/T':>10} {'rel. err':>10} "
      f"{'tau/T-1/2':>12} {'rel. err':>10} {'tau/T-1/2+T/12tau':>18} {'rel. err':>10}")
for T, tau in [(20e-9, 50e-6), (20e-9, 100e-9)]:
    x = T / tau
    M = M_exact(T, tau)
    a0 = 1 / x                  # Jordanov & Knoll's tau/T
    a1 = 1 / x - 0.5            # Appendix A, Eq. A.4
    a2 = 1 / x - 0.5 + x / 12   # next term of the series
    print(f"{1/x:8.0f} {M:16.9f} {a0:10.1f} {(a0-M)/M:10.2e} "
          f"{a1:12.4f} {(a1-M)/M:10.2e} {a2:18.6f} {(a2-M)/M:10.2e}")

print("\nPart 2 - running sum + M x input, applied to A b^n (A = 1)")
T, tau = 20e-9, 50e-6
b = np.exp(-T / tau)
n = np.arange(20000)                         # 400 us, 8 decay constants
v = b ** n
M = M_exact(T, tau)
for label, Mt in [("exact M", M), ("M = tau/T", tau / T)]:
    q = np.cumsum(v) + Mt * v                # p[n] + M d[n], with d replaced by v
    step = M + 1                             # Eq. A.2: height A(M+1)
    dev = (q - step) / step
    print(f"  {label:10s}: q[0] = {q[0]:.6f}, q[end] = {q[-1]:.6f}, "
          f"A(M+1) = {step:.6f}, max |relative deviation| = {np.max(np.abs(dev)):.2e}")
