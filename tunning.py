import numpy as np
import allantools

def allan_params(signal, fs):
    tau, adev, _, _ = allantools.oadev(np.asarray(signal, float), rate=fs, data_type="freq", taus="all")
    slopes = np.gradient(np.log(adev), np.log(tau))

    # White noise (slope ≈ -0.5) -> N^2 (continuous-time PSD)
    mw = (slopes >= -0.65) & (slopes <= -0.35)
    n_sq = (np.median(adev[mw] * np.sqrt(tau[mw])) ** 2) if np.any(mw) else np.nan

    # Rate random walk (slope ≈ +0.5) -> rrw variance
    mr = (slopes >= 0.35) & (slopes <= 0.65)
    rrw = (np.median(adev[mr] / np.sqrt(tau[mr] / 3.0)) ** 2) if np.any(mr) else np.nan

    return n_sq, rrw

def q_from_imu(ax, ay, gz, fs, rrw_floor=1e-12):
    (n1, r1), (n2, r2), (n3, r3) = [allan_params(s, fs) for s in (ax, ay, gz)]
    safe = lambda v: v if np.isfinite(v) else rrw_floor
    
    # Returns continuous-time parameters (q1..q6) for C++ EKF runtime scaling
    return safe(n1), safe(n2), safe(n3), safe(r1), safe(r2), safe(r3)