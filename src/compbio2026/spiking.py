"""Driving biophysical neurons with the SHD spike trains, as spikes.

SHD is the output of the cochlear nucleus: 700 bushy cells, one spike train
each. The obvious next model is the next synapse — those spike trains arriving
at a downstream population through real synapses. That is what this module
builds, and it deliberately does **not** filter, pool, compress or rate-code the
input first. Every one of those steps is a modelling decision that a cochlea and
its targets do not make, and each one quietly removes the thing the exercise is
about.

The construction
----------------
Each of the 700 SHD channels is one presynaptic compartment whose membrane
potential is **clamped** to a spike waveform read off the raw event list: at
rest otherwise, driven to ``v_spike`` for ``width_ms`` at each recorded spike
time. Those compartments connect to a downstream population of Hodgkin–Huxley
cells through :class:`jaxley.synapses.IonotropicSynapse`, so the drive is a
**conductance** with a reversal potential, not an injected current — divisive as
well as additive, and it saturates as the postsynaptic cell approaches
``e_syn``. Nothing about the stimulus is compressed: the simulation runs in real
time at a real timestep.

Clamping the presynaptic voltage rather than simulating a presynaptic cell is
the one liberty taken here, and it is the right one: SHD *is* the spike train,
so a model of the cell that produced it would be a fiction inserted between the
data and the circuit.

Cost, measured on a laptop CPU: a few seconds to build a 700-to-24 network, and
one to two seconds per trial of 700 ms at ``dt = 0.1 ms`` — so a 160-trial
dataset takes about five minutes. Gradients through the simulation cost roughly
ten times that and need ``checkpoint_lengths``, which is the whole reason
notebook 07 offers a reduced pipeline for training.

JAXley is imported lazily, so importing :mod:`compbio2026` in the analysis
environment does not require the Route B stack.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Input: raw SHD events -> a presynaptic voltage waveform
# --------------------------------------------------------------------------

def spike_drive(
    shd,
    trials,
    dt_ms: float = 0.1,
    t_max_ms: float = 700.0,
    n_channels: int = 700,
    v_rest: float = -70.0,
    v_spike: float = 20.0,
    width_ms: float = 1.0,
) -> np.ndarray:
    """Presynaptic voltage clamp waveforms, ``(n_trials, n_channels, n_steps)``.

    Built from :func:`compbio2026.data.events` — the raw event list, not a
    design matrix. The only quantisation is onto the simulation grid, and at
    ``dt_ms = 0.1`` that is far finer than the ~1.2 ms refractory floor of the
    bushy-cell model, so no spike is lost or merged.

    ``width_ms`` is the duration of the clamped depolarisation. It sets how long
    the synapse sees suprathreshold presynaptic voltage and therefore how much
    transmitter one spike releases; 1 ms is a reasonable action-potential width
    and is worth varying to see how little the result depends on it.

    Memory is ``n_trials * n_channels * n_steps`` floats — 157 MB for 8 trials
    of 700 ms at ``dt = 0.1 ms``. Build it per batch, not for the whole dataset.
    """
    from .data import events

    trials = np.atleast_1d(np.asarray(trials))
    n_steps = int(round(t_max_ms / dt_ms))
    n_wide = max(int(round(width_ms / dt_ms)), 1)

    v = np.full((len(trials), n_channels, n_steps), v_rest, dtype=np.float32)
    for row, i in enumerate(trials):
        ev = events(shd, int(i), t_max_ms=t_max_ms, n_channels=n_channels)
        u = ev[:, 0].astype(int)
        b = (ev[:, 1] / dt_ms).astype(int)
        for k in range(n_wide):
            v[row, u, np.minimum(b + k, n_steps - 1)] = v_spike
    return v


# --------------------------------------------------------------------------
# Connectivity
# --------------------------------------------------------------------------

def tonotopic_connectivity(
    n_in: int,
    n_out: int,
    fan_in: int = 60,
    width: float = 35.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Boolean ``(n_in, n_out)`` matrix: cell *j* samples channels near its centre.

    Downstream cell *j* is assigned a best frequency by placing its centre at
    ``j / n_out`` of the way along the channel axis, then draws ``fan_in``
    presynaptic channels from a Gaussian of standard deviation ``width``
    channels around it. This is the biological version of what pooling was
    standing in for: convergence onto a cell with a receptive field, rather than
    an average computed outside the model.

    ``width`` controls how sharply tuned the layer is before any learning.
    Setting it very large gives untuned cells that see the whole cochlea, which
    is the control worth running.
    """
    rng = rng or np.random.default_rng(0)
    M = np.zeros((n_in, n_out), dtype=bool)
    channels = np.arange(n_in)
    for j, centre in enumerate(np.linspace(0, n_in - 1, n_out)):
        p = np.exp(-0.5 * ((channels - centre) / width) ** 2)
        p /= p.sum()
        M[rng.choice(n_in, size=fan_in, replace=False, p=p), j] = True
    return M


def random_connectivity(
    n_in: int, n_out: int, fan_in: int = 60, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Boolean ``(n_in, n_out)``, each cell sampling channels uniformly.

    The control for :func:`tonotopic_connectivity`. If the downstream layer does
    the same job with random convergence, the tonotopic wiring was decoration.
    """
    rng = rng or np.random.default_rng(0)
    M = np.zeros((n_in, n_out), dtype=bool)
    for j in range(n_out):
        M[rng.choice(n_in, size=fan_in, replace=False), j] = True
    return M


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

#: Conductance densities that give an :class:`~jaxley.channels.AdEx` cell a
#: ~20 ms membrane time constant on the 10 x 10 um compartment used here.
#:
#: JAXley's AdEx defaults are written for a much smaller compartment: ``g_L``
#: of 10 S/cm² against a capacitance of 1 uF/cm² gives ``tau_m = C/g_L`` of
#: 0.1 microseconds, so the cell integrates nothing and never fires however
#: hard you drive it. ``g_L = 5e-5`` puts ``tau_m`` at 20 ms; ``a`` keeps the
#: published ratio ``a/g_L = 0.2``. Pass these as ``channel_params``.
ADEX_PARAMS = {
    "AdEx_g_L": 5e-5,     # S/cm², tau_m = C/g_L = 20 ms
    "AdEx_a": 1e-5,       # subthreshold adaptation
    "AdEx_b": 0.0,        # spike-triggered adaptation; raise for spike-frequency adaptation
    "AdEx_tau_w": 100.0,  # ms
}


def build_layer(
    connectivity: np.ndarray,
    channel=None,
    channel_params: dict | None = None,
    tau_syn_ms: float = 5.0,
    e_syn_mV: float = 0.0,
    v_th_mV: float = -35.0,
    delta_mV: float = 2.0,
    radius_um: float = 10.0,
    length_um: float = 10.0,
):
    """A clamped input population wired to a spiking population.

    Returns the :class:`jaxley.Network`. Cells ``0 .. n_in-1`` are the input
    compartments (voltage-clamped, :class:`~jaxley.channels.Leak` only); cells
    ``n_in .. n_in+n_out-1`` carry the downstream channel and are the ones
    recorded.

    ``channel``
        Channel inserted into the downstream population; defaults to
        :class:`~jaxley.channels.HH`. :class:`~jaxley.channels.AdEx` gives an
        adaptive exponential integrate-and-fire cell that simulates about
        fifteen times faster, at the cost of gradients that are zero after
        every spike — fine for running the model, useless for training through
        it. With AdEx, pass ``channel_params=ADEX_PARAMS`` and detect spikes
        with a threshold below its 0 mV cap, for instance ``-20.0``.
    ``channel_params``
        Parameters applied with ``net.set`` after insertion.

    ``delta_mV`` is the steepness of the presynaptic sigmoid that gates
    transmitter release. The JAXley default of 10 mV leaves a synapse ~3 % open
    at a resting presynaptic voltage of -70 mV, which with a fan-in of 60 is a
    large tonic conductance that nothing in the data put there. 2 mV makes the
    synapse effectively silent between spikes, which is what you want when the
    presynaptic voltage is a clamped stand-in for a spike rather than a
    simulated membrane.

    ``tau_syn_ms`` is the decay of the synaptic conductance. It is the only
    filtering in the model, and unlike a preprocessing kernel it is a property
    of the synapse rather than of the analysis.
    """
    import jaxley as jx
    from jaxley.channels import HH, Leak
    from jaxley.connect import connectivity_matrix_connect
    from jaxley.synapses import IonotropicSynapse

    channel = channel if channel is not None else HH()

    n_in, n_out = connectivity.shape
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    cell = jx.Cell(branch, parents=[-1])
    # Reusing one Cell object is ~10x faster to construct than building n_in+n_out
    # of them, and jaxley copies what it needs.
    net = jx.Network([cell] * (n_in + n_out))

    pre = net.cell(list(range(n_in)))
    post = net.cell(list(range(n_in, n_in + n_out)))
    pre.insert(Leak())
    post.insert(channel)
    connectivity_matrix_connect(pre, post, IonotropicSynapse(), connectivity)

    net.set("radius", radius_um)
    net.set("length", length_um)
    net.set("IonotropicSynapse_k_minus", 1.0 / tau_syn_ms)
    net.set("IonotropicSynapse_e_syn", e_syn_mV)
    net.set("IonotropicSynapse_v_th", v_th_mV)
    net.set("IonotropicSynapse_delta", delta_mV)
    for key, value in (channel_params or {}).items():
        net.set(key, value)
    post.record("v", verbose=False)
    return net


def make_simulator(net, n_in: int, dt_ms: float = 0.1, checkpoints=None):
    """Return ``sim(g_syn, v_clamp)`` for one trial, differentiable in ``g_syn``.

    ``g_syn`` is a ``(n_edges,)`` vector of maximal synaptic conductances in µS,
    one per connection, in the order of ``net.edges``. ``v_clamp`` is one trial
    of :func:`spike_drive`, ``(n_in, n_steps)``. The return is the recorded
    membrane potential of the downstream cells, ``(n_out, n_steps + 1)`` in mV.

    ``jax.vmap`` it over trials to run a batch, and ``jax.jit`` the result.
    Pass ``checkpoints`` (for example ``[80, 100]`` for 8000 steps, the two
    factors multiplying to the step count) only when taking gradients: it trades
    recomputation for the memory that reverse-mode differentiation through
    thousands of timesteps would otherwise need.
    """
    import jaxley as jx

    n_edges = len(net.edges)
    edge_view = net.select(edges=np.arange(n_edges))
    pre_view = net.cell(list(range(n_in)))

    # Build the index bookkeeping once, outside any trace. JAXley indexes numpy
    # arrays with these, so they have to stay concrete; calling `data_set` or
    # `data_clamp` on a traced argument makes them tracers and `jx.integrate`
    # then fails inside `jit`. Only the values are substituted per call.
    template = edge_view.data_set("IonotropicSynapse_gS", 0.0, None)
    indices, key = template[0]["indices"], template[0]["key"]
    assert indices.shape[0] == n_edges, "not every edge carries IonotropicSynapse_gS"
    clamp_names, _, clamp_inds = pre_view.data_clamp("v", np.zeros((n_in, 1), np.float32))

    def sim(g_syn, v_clamp):
        param_state = [{"indices": indices, "key": key, "val": g_syn}]
        clamps = (clamp_names, [v_clamp], clamp_inds)
        n_steps = v_clamp.shape[-1]
        return jx.integrate(
            net,
            param_state=param_state,
            data_clamps=clamps,
            delta_t=dt_ms,
            t_max=(n_steps - 1) * dt_ms,
            checkpoint_lengths=checkpoints,
        )

    return sim


# --------------------------------------------------------------------------
# Output: voltages -> spikes -> something the rest of the toolkit can read
# --------------------------------------------------------------------------

def detect_spikes(v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean array marking upward crossings of ``threshold``, same leading shape.

    Crossings rather than "voltage is high", so one action potential counts once
    however long it stays depolarised. The returned array has one fewer sample
    along time than ``v``.

    The default suits Hodgkin–Huxley, whose spikes overshoot 0 mV. AdEx never
    does: it caps the membrane at ``v_threshold`` (0 mV) and resets, so pass
    something below that cap and above ``v_T``, for instance ``-20.0``.
    """
    v = np.asarray(v)
    return (v[..., 1:] > threshold) & (v[..., :-1] <= threshold)


def spike_counts(v: np.ndarray, dt_ms: float, bin_ms: float = 20.0,
                 threshold: float = 0.0) -> np.ndarray:
    """Bin the downstream spikes into ``(n_trials, n_cells, n_bins)`` counts.

    The same shape as :func:`compbio2026.data.build_design_matrix` returns for
    the input layer, on purpose: the point of simulating a second layer is to
    put it through the same geometry and decoding analysis as the first, with
    the same code and the same controls.
    """
    crossings = detect_spikes(v, threshold)
    n_steps = crossings.shape[-1]
    per_bin = max(int(round(bin_ms / dt_ms)), 1)
    n_bins = n_steps // per_bin
    trimmed = crossings[..., : n_bins * per_bin]
    return trimmed.reshape(crossings.shape[:-1] + (n_bins, per_bin)).sum(-1).astype(np.float32)


def firing_rates(v: np.ndarray, dt_ms: float, threshold: float = 0.0) -> np.ndarray:
    """Mean firing rate per cell per trial, in spikes/second."""
    crossings = detect_spikes(v, threshold)
    duration_s = crossings.shape[-1] * dt_ms / 1e3
    return crossings.sum(-1) / duration_s
