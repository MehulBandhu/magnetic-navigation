"""The corrector: a small causal 1-D convolutional network that predicts the residual the
linear Tolles-Lawson model leaves, from what the vehicle logs anyway.

Inputs at each step (F = 46 channels): attitude (roll, pitch, sin and cos of yaw), rates (3),
the 18 Tolles-Lawson regressors divided by B0 (the direction cosines, their products and
their products with the derivatives), per-motor current and RPM padded to 8 motors, battery
voltage and current, throttle, the servo flag, and the linear-model residual itself. The
model sees a window of the last `window` steps; the residual channel is blanked for the last
`gap` steps, so the network cannot copy the current value and has to carry the level forward
from telemetry and from what the residual did a couple of seconds ago. That is what a
navigator has: the residual is only known against the map, at the position it estimated.

Output: the residual at the current step, nT. Architecture: a stack of dilated causal
convolutions (receptive field 125 steps) with residual connections, under 200k parameters.
"""
import json

import numpy as np
import torch
import torch.nn as nn

from problem1.checks import B0

from .platform import MAX_MOTORS

RESIDUAL_SCALE_NT = 10.0     # residuals are stored in units of 10 nT
LAG_TAUS_S = (0.03, 0.1, 0.3, 1.0, 3.0)   # the eddy regressors are also given filtered at these time constants
BASE_FEATURE_NAMES = (["roll", "pitch", "sin_yaw", "cos_yaw", "roll_rate", "pitch_rate", "yaw_rate"]
                      + [f"tl_{i}" for i in range(18)]
                      + [f"eddy_tau{tau}_{i}" for tau in LAG_TAUS_S for i in range(9)]
                      + [f"motor_current_{i}" for i in range(MAX_MOTORS)] + [f"motor_rpm_{i}" for i in range(MAX_MOTORS)]
                      + ["battery_voltage", "battery_current", "throttle", "servo"])
# two channels are added per window: a flag saying which residual is shown (0: after the
# linear model, 1: after the lag model) and the residual itself
FEATURE_NAMES = BASE_FEATURE_NAMES + ["after_lag_model", "residual"]
N_BASE = len(BASE_FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def lagged_eddy(u, udot, dt_s, taus=LAG_TAUS_S):
    """(T, 9 * len(taus)): u_i times the first-order lag of udot_j (edge_cases.lag) at each
    time constant. The same information as the rates, filtered, so the network does not have
    to learn a long convolution to see a lagged eddy current."""
    from problem1.edge_cases import lag
    cols = []
    for tau in taus:
        ld = lag(udot, tau, dt_s)
        cols += [u[:, i] * ld[:, j] for i in range(3) for j in range(3)]
    return np.stack(cols, 1)


def features(flight_est, A, telemetry, u, udot):
    """(T, N_BASE) float32 feature matrix, everything but the residual. flight_est: the
    attitude the estimator has (true or IMU); A: the (T, 18) regressor built from it;
    telemetry: from platform.interference; u, udot: the direction cosines and their
    derivative the regressor was built from."""
    cols = [flight_est["roll_rad"], flight_est["pitch_rad"], np.sin(flight_est["yaw_rad"]), np.cos(flight_est["yaw_rad"]),
            flight_est["roll_rate"], flight_est["pitch_rate"], flight_est["yaw_rate"]]
    X = np.column_stack(cols + [A / B0, lagged_eddy(u, udot, flight_est["dt_s"]), telemetry["motor_current_A"] / 30.0, telemetry["motor_rpm"] / 10000.0,
                                telemetry["battery_voltage_V"][:, None] / 25.0, telemetry["battery_current_A"][:, None] / 100.0,
                                telemetry["throttle"][:, None], telemetry["servo"][:, None].astype(float)])
    return X.astype(np.float32)


def windows(X, residual_nT, after_lag, t_idx, window, gap, min_scale_nT=0.5):
    """(len(t_idx), N_FEATURES, window) input tensors ending at each t in t_idx, and the level
    (N,) and scale (N,) in nT of the residual over the part of each window that is shown.
    X is the (T, N_BASE) feature matrix, residual_nT the (T,) residual the corrector is shown
    (after the linear model, or after the lag model when after_lag is set). The flight is
    edge padded at the front; the residual is taken relative to its level (a bias the
    navigator absorbs) and divided by its scale, then zeroed over the last `gap` steps. The
    model's output is in the same units: residual = level + scale * output."""
    T = X.shape[0]
    idx = t_idx[:, None] - np.arange(window)[::-1][None, :]
    idx = np.clip(idx, 0, T - 1)
    N = len(t_idx)
    W = np.empty((N, N_FEATURES, window), dtype=np.float32)
    W[:, :N_BASE, :] = X[idx].transpose(0, 2, 1)
    W[:, N_BASE, :] = 1.0 if after_lag else 0.0
    shown = residual_nT[idx[:, :window - gap]]
    level = shown.mean(1)
    scale = np.maximum(shown.std(1), min_scale_nT)
    W[:, -1, :] = (residual_nT[idx] - level[:, None]) / scale[:, None]
    W[:, -1, window - gap:] = 0.0
    return W, level, scale


class CausalConv(nn.Module):
    """left-padded convolution whose last output sees the last input, with an optional stride
    that shortens the sequence (the padding is adjusted at run time so the alignment holds)"""
    def __init__(self, n_in, n_out, kernel, dilation=1, stride=1):
        super().__init__()
        self.kernel, self.dilation, self.stride = kernel, dilation, stride
        self.conv = nn.Conv1d(n_in, n_out, kernel, dilation=dilation, stride=stride)

    def forward(self, x):
        L = x.shape[-1]
        pad = (self.kernel - 1) * self.dilation + (-(L - 1)) % self.stride
        return self.conv(nn.functional.pad(x, (pad, 0)))


class CausalBlock(nn.Module):
    def __init__(self, width, kernel, dilation):
        super().__init__()
        self.conv = CausalConv(width, width, kernel, dilation)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(x))


class Corrector(nn.Module):
    """two strided causal convolutions bring the 10 Hz window down to 2.5 Hz, then dilated
    causal blocks with residual connections; the head reads the last step and predicts the
    residual there in units of RESIDUAL_SCALE_NT"""
    def __init__(self, n_in=N_FEATURES, width=64, kernel=5, dilations=(1, 2, 4, 8, 16), window=200, gap=20,
                 mean=None, std=None, strides=(2, 2), ridge_dim=0):
        super().__init__()
        self.window, self.gap, self.ridge_dim = window, gap, ridge_dim
        self.total_stride = int(np.prod(strides)) if len(strides) else 1
        self.register_buffer("mean", torch.zeros(n_in) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(n_in) if std is None else torch.as_tensor(std, dtype=torch.float32))
        front, w_in = [], n_in
        for s in strides:
            front += [CausalConv(w_in, width, kernel, 1, s), nn.GELU()]
            w_in = width
        self.front = nn.Sequential(*front) if front else nn.Conv1d(n_in, width, 1)
        self.blocks = nn.Sequential(*[CausalBlock(width, kernel, d) for d in dilations])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        if ridge_dim:
            # in-context least squares: a learned basis phi_t of the inputs, fitted to the shown
            # part of the residual within the window by ridge regression in closed form, then
            # read at the current step. The network learns the basis; the fit is per window,
            # which is how the vehicle is identified from its own recent residual.
            self.basis = nn.Linear(width, ridge_dim)
            self.log_lambda = nn.Parameter(torch.zeros(()))
        self.config = dict(n_in=n_in, width=width, kernel=kernel, dilations=list(dilations), window=window, gap=gap,
                           strides=list(strides), ridge_dim=ridge_dim)

    def forward(self, x):
        # x: (N, F, window); the residual channel's zeros in the gap stay zero after
        # normalisation only if its mean is zero, which the training script enforces
        r = x[:, -1, :]                                          # the shown residual, normalised, zero in the gap
        x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        h = self.blocks(self.front(x))                          # (N, width, L_out), last step aligned with the input's
        out = self.head(h[:, :, -1]).squeeze(-1)
        if not self.ridge_dim:
            return out
        L, L_out = x.shape[-1], h.shape[-1]
        idx = L - 1 - (L_out - 1 - torch.arange(L_out, device=x.device)) * self.total_stride
        keep = idx >= 0
        shown = keep & (idx < L - self.gap)
        phi = self.basis(h.transpose(1, 2))                     # (N, L_out, k)
        target = r[:, idx.clamp(min=0)] * shown.float()          # (N, L_out)
        P = phi * shown.float()[None, :, None]
        G = P.transpose(1, 2) @ P + torch.exp(self.log_lambda) * torch.eye(self.ridge_dim, device=x.device)
        w = torch.linalg.solve(G, (P.transpose(1, 2) @ target[:, :, None]))   # (N, k, 1)
        return out + (phi[:, -1, :, None] * w).sum((1, 2))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def predict(model, X, residual_nT, after_lag=False, batch=512):
    """the corrector's prediction (T,) nT of the residual along a whole flight"""
    model.eval()
    T = X.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for s in range(0, T, batch):
        t_idx = np.arange(s, min(T, s + batch))
        w, level, scale = windows(X, residual_nT, after_lag, t_idx, model.window, model.gap)
        out[s:s + len(t_idx)] = model(torch.from_numpy(w)).numpy() * scale + level
    return out


def save(model, path_pt, extra=None):
    torch.save(model.state_dict(), path_pt)
    json.dump(dict(config=model.config, **(extra or {})), open(path_pt.replace(".pt", ".json"), "w"), indent=1)


def load(path_pt="runs/corrector.pt"):
    c = json.load(open(path_pt.replace(".pt", ".json")))["config"]
    model = Corrector(c["n_in"], c["width"], c["kernel"], tuple(c["dilations"]), c["window"], c["gap"], strides=tuple(c["strides"]),
                      ridge_dim=c.get("ridge_dim", 0))
    model.load_state_dict(torch.load(path_pt, map_location="cpu"))
    model.eval()
    return model
