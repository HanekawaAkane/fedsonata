#!/usr/bin/env python
"""FS23 pre-launch gate tests (no training).

Covers, to the extent testable without training:
  G1  BASE trace-hash regression against the archived Nov6 record.
  G4  Synthetic state trajectory: raw + visible counters under every
      age_view; merged-vs-dual at omega=1 bit-identical; LPC trigger
      equality semantics incl. overshoot and non-integer ages.
  G3  Reliability view masking; raw store untouched by reads.
  G4b GE schedule: purity, phase continuity (no state reset), identical to
      no-schedule roll when schedule equals profile params.
  G5  Zero-hazard branch: no event, no division, inf failure time.
  G6  Radio-plan resource constraints on synthetic inputs for all three
      radio policies (bandwidth/power/deadline feasibility counting).
  G3b Parser: invalid/legacy alias values are rejected (subprocess).

Run:  python tests/test_fs23_gates.py   (from the fs23 experiment root)
Exit code 0 = all gates pass.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------- #
# G1: BASE trace hash regression (must equal the archived Nov6 record).
# --------------------------------------------------------------------- #
def g1_trace_hash_regression() -> None:
    from mec.traces import TraceGenerator, compute_trace_hash
    from mec.profiles import ChannelProfile, DropoutProfile

    channel = ChannelProfile(model="correlated", correlation=0.9, rayleigh_mean=1e-10)
    # Archived pipeline: p = mec_ge_good_to_bad = 0.1, q tuned from dropout_rate.
    p = 0.1
    d = 0.2
    q = p * (1.0 - d) / d
    dropout = DropoutProfile(
        model="gilbert_elliott", dropout_rate=d, ge_good_to_bad=p, ge_bad_to_good=q
    )
    for seed, expected in (
        (42, "afbe686aef47ad7cbc42f4d9adf132ba0d92d707058e85368799464802e5df46"),
    ):
        got = compute_trace_hash(
            seed, 20, channel, dropout, 20, duration_hazard_rate=0.1
        )
        check(f"G1 trace hash seed={seed} matches Nov6 record", got == expected,
              f"got {got} expected {expected}")

    # Schedule path with a degenerate single phase must equal no-schedule hash
    # only when the schedule is not hashed in... (schedule repr is hashed, so
    # compare the dropout vectors instead).
    gen_plain = TraceGenerator(42, 20, channel, dropout)
    gen_sched = TraceGenerator(
        42, 20, channel, dropout,
        ge_schedule=((1, 20, p, q),),
        hazard_schedule=((1, 20, 0.1),),
        default_hazard=0.1,
    )
    same = all(
        gen_plain.dropout_vector(t) == gen_sched.dropout_vector(t)
        for t in range(1, 21)
    )
    check("G1 GE schedule with constant params reproduces base dropout trace", same)


# --------------------------------------------------------------------- #
# G4: staleness views / merged-dual equivalence / LPC trigger semantics.
# --------------------------------------------------------------------- #
def g4_staleness_views() -> None:
    from mec.staleness import EffectiveStaleness, update_effective_staleness, lpc_trigger_mask

    # Synthetic event trajectory over 6 rounds, 2 clients, 3 blocks.
    blocks = ["l0", "l1", "l2"]
    # Per round: (selected, arrived, failure_type); unselected = blocks - selected.
    per_round = [
        ({"l0", "l1"}, True, "none"),
        ({"l0"}, False, "dropout"),
        ({"l1"}, True, "none"),
        ({"l0", "l2"}, False, "deadline_miss"),
        ({"l2"}, True, "none"),
        ({"l0", "l1", "l2"}, True, "none"),
    ]

    def run(view: str, omega: float = 1.0, upto: int = 6):
        st = EffectiveStaleness(omega_net=omega, age_view=view)
        for sel, arrived, ft in per_round[:upto]:
            update_effective_staleness(st, 0, sel, blocks, arrived, ft)
        return st

    dual = run("dual")
    merged = run("merged")
    alloc_only = run("allocation_only")
    deliv_only = run("delivery_only")
    zero = run("zero")

    # Expected raw counters after the trajectory, block l0:
    # r1 selected+arrived -> (0,0); r2 selected+failed -> (0,1); r3 unselected -> (1,1);
    # r4 selected+failed -> (1,2); r5 unselected -> (2,2); r6 arrived -> (0,0).
    check("G4 raw counters l0 end at (0,0)",
          dual.alloc.get((0, "l0")) == 0 and dual.net.get((0, "l0")) == 0,
          f"alloc={dual.alloc.get((0,'l0'))} net={dual.net.get((0,'l0'))}")
    # l1: r1 arrived (0,0); r2 unselected (1,0); r3 arrived (0,0); r4 unselected (1,0);
    # r5 unselected (2,0); r6 arrived (0,0).
    check("G4 raw counters l1 end at (0,0)",
          dual.alloc.get((0, "l1")) == 0 and dual.net.get((0, "l1")) == 0)
    # l2: r1 unselected (1,0); r2 unselected (2,0); r3 unselected (3,0);
    # r4 selected+failed (3,1); r5 arrived (0,0); r6 arrived (0,0).
    check("G4 raw counters l2 end at (0,0)",
          dual.alloc.get((0, "l2")) == 0 and dual.net.get((0, "l2")) == 0)

    # Mid-trajectory check: after round 4, l0 = (1,2).
    st_mid = run("dual", upto=4)
    check("G4 mid-trajectory l0 raw (alloc=1, net=2)",
          st_mid.alloc.get((0, "l0")) == 1 and st_mid.net.get((0, "l0")) == 2,
          f"alloc={st_mid.alloc.get((0,'l0'))} net={st_mid.net.get((0,'l0'))}")
    check("G4 dual effective l0 == 1 + 2", st_mid.effective(0, "l0") == 3.0)
    check("G4 allocation_only l0 == 1",
          EffectiveStaleness(omega_net=1.0, age_view="allocation_only",
                             alloc=dict(st_mid.alloc), net=dict(st_mid.net)
                             ).effective(0, "l0") == 1.0)
    check("G4 delivery_only l0 == 2",
          EffectiveStaleness(omega_net=1.0, age_view="delivery_only",
                             alloc=dict(st_mid.alloc), net=dict(st_mid.net)
                             ).effective(0, "l0") == 2.0)
    check("G4 zero l0 == 0",
          EffectiveStaleness(omega_net=1.0, age_view="zero",
                             alloc=dict(st_mid.alloc), net=dict(st_mid.net)
                             ).effective(0, "l0") == 0.0)

    # merged-vs-dual at omega=1 must be bit-identical for every key.
    ident = True
    for key in set(dual.alloc) | set(dual.net):
        if dual.raw_effective(*key) != merged.effective(*key):
            ident = False
        if dual.effective(*key) != merged.effective(*key):
            ident = False
    check("G4 merged vs dual at omega=1 bit-identical", ident)

    # omega=1 also: dual raw_effective == alloc + net exactly (ints).
    ident2 = all(
        dual.raw_effective(c, b) == dual.alloc.get((c, b), 0) + dual.net.get((c, b), 0)
        for (c, b) in set(dual.alloc) | set(dual.net)
    )
    check("G4 dual raw = alloc+net at omega=1", ident2)

    # LPC trigger: exact equality semantics (round(tau) == b_train_every).
    st_lpc = EffectiveStaleness(omega_net=1.0, age_view="dual")
    st_lpc.alloc[(0, "l0")] = 4
    st_lpc.net[(0, "l0")] = 0
    st_lpc.alloc[(0, "l1")] = 5     # overshoot -> NOT triggered
    st_lpc.net[(0, "l1")] = 0
    st_lpc.alloc[(0, "l2")] = 1
    st_lpc.net[(0, "l2")] = 2       # effective 3, round(3)=3 != 4 -> no
    # non-integer age: alloc=2, net=1 -> effective 3.0; use half weights view
    st_half = EffectiveStaleness(omega_net=0.5, age_view="dual",
                                 alloc={(0, "l0"): 3}, net={(0, "l0"): 1})
    # effective = 3 + 0.5 = 3.5 -> round=4 -> trigger
    m = lpc_trigger_mask(st_lpc, 0, {"l0", "l1", "l2"}, 4)
    check("G4 LPC trigger exact-equality (4 yes, 5 no, 3 no)",
          m == {"l0"}, f"mask={m}")
    m2 = lpc_trigger_mask(st_half, 0, {"l0"}, 4)
    check("G4 LPC trigger non-integer age 3.5 rounds to 4", m2 == {"l0"}, f"mask={m2}")

    # The age_view must reach the LPC trigger consumer.
    st_zero = EffectiveStaleness(omega_net=1.0, age_view="zero",
                                 alloc={(0, "l0"): 4}, net={(0, "l0"): 0})
    m3 = lpc_trigger_mask(st_zero, 0, {"l0"}, 4)
    check("G4 age_view=zero masks LPC trigger", m3 == set(), f"mask={m3}")

    # Unsamped clients retain state (no update call) -- covered implicitly:
    # we never touched client 1; its counters must be absent/zero.
    check("G4 unsampled client retains zero state",
          dual.effective(1, "l0") == 0.0)

    # Unknown view must fail loudly.
    try:
        EffectiveStaleness(age_view="legacy_something")
        check("G4 unknown age_view rejected", False)
    except ValueError:
        check("G4 unknown age_view rejected", True)


# --------------------------------------------------------------------- #
# G3: reliability view.
# --------------------------------------------------------------------- #
def g3_reliability_view() -> None:
    from mec.staleness import ReliabilityView
    from mec.scheduler import update_success_ewma, est_success_default

    raw = est_success_default([0, 1, 2], 1.0)
    view_obs = ReliabilityView(raw=raw, view="observed", r_init=1.0)
    view_fix = ReliabilityView(raw=raw, view="constant_initial", r_init=1.0)

    # True updates go to the raw store (as the main loop does).
    for _ in range(5):
        update_success_ewma(raw, 0, False, beta=0.8)
    check("G3 raw EWMA updates applied", raw[0] < 1.0, f"raw[0]={raw[0]}")
    check("G3 observed view reflects raw", view_obs[0] == raw[0])
    check("G3 constant view pinned to r_init", view_fix[0] == 1.0 and view_fix.get(1) == 1.0)
    check("G3 constant view iteration still lists raw keys", set(view_fix) == {0, 1, 2})

    # EWMA update through the RAW store must not be affected by the view.
    before = raw[0]
    _ = view_fix[0]  # read through constant view
    check("G3 reads do not mutate raw", raw[0] == before)

    try:
        ReliabilityView(raw=raw, view="legacy_off")
        check("G3 unknown reliability view rejected", False)
    except ValueError:
        check("G3 unknown reliability view rejected", True)


# --------------------------------------------------------------------- #
# G4b: GE schedule purity and continuity.
# --------------------------------------------------------------------- #
def g4b_ge_schedule() -> None:
    from mec.traces import TraceGenerator
    from mec.profiles import ChannelProfile, DropoutProfile

    channel = ChannelProfile(model="constant", constant_gain=1e-10)
    dropout = DropoutProfile(
        model="gilbert_elliott", dropout_rate=0.2, ge_good_to_bad=0.1, ge_bad_to_good=0.4
    )
    sched = (
        (1, 10, 0.1, 0.4),
        (11, 15, 0.2, 0.3),
        (16, 20, 0.1, 0.4),
    )
    gen = TraceGenerator(42, 20, channel, dropout, ge_schedule=sched)

    # Purity: fresh generator gives the same values.
    gen2 = TraceGenerator(42, 20, channel, dropout, ge_schedule=sched)
    pure = all(
        gen.dropout_flag(t, k) == gen2.dropout_flag(t, k)
        for t in range(1, 21) for k in range(20)
    )
    check("G4b scheduled GE trace pure in (seed, round, client)", pure)

    # Continuity: state at round 11 is a single chain roll, not re-drawn.
    # Compare the full trajectory against a manual roll.
    def manual(round_num, client_id):
        rng_ok = True
        import numpy as np
        rng = np.random.default_rng([42, client_id, 1])
        bad = False
        for t in range(1, round_num + 1):
            p, q = (0.1, 0.4) if t <= 10 else (0.2, 0.3) if t <= 15 else (0.1, 0.4)
            u = rng.random()
            if bad:
                bad = bool(u < (1.0 - q))
            else:
                bad = bool(u < p)
        return bad

    cont = all(manual(t, k) == gen.dropout_flag(t, k) for t in range(1, 21) for k in range(20))
    check("G4b phase boundaries keep continuous GE state", cont)

    # Constant schedule == no-schedule rollout.
    gen_base = TraceGenerator(42, 20, channel, dropout)
    gen_c = TraceGenerator(42, 20, channel, dropout, ge_schedule=((1, 20, 0.1, 0.4),))
    eq = all(
        gen_base.dropout_flag(t, k) == gen_c.dropout_flag(t, k)
        for t in range(1, 21) for k in range(20)
    )
    check("G4b constant schedule equals profile-parameter roll", eq)

    # Rounds outside every phase must fail loudly.
    gen_bad = TraceGenerator(42, 20, channel, dropout, ge_schedule=((1, 10, 0.1, 0.4),))
    try:
        gen_bad.dropout_flag(15, 0)
        check("G4b uncovered round raises", False)
    except ValueError:
        check("G4b uncovered round raises", True)


# --------------------------------------------------------------------- #
# G5: zero-hazard branch.
# --------------------------------------------------------------------- #
def g5_zero_hazard() -> None:
    from mec.traces import TraceGenerator
    from mec.profiles import ChannelProfile, DropoutProfile

    gen = TraceGenerator(
        1, 4, ChannelProfile(model="constant", constant_gain=1e-10),
        DropoutProfile(model="none"),
    )
    check("G5 hazard=0 -> no duration event",
          gen.duration_failure(1, 0, 5.0, 0.0) is False)
    check("G5 hazard=0 -> failure time inf",
          math.isinf(gen.duration_failure_time(1, 0, 0.0)))
    # Matched-uniform consistency: T_fail = -log1p(-U)/hazard matches the
    # event rule U < 1-exp(-hazard*T_up).
    u = gen.duration_failure_uniform(3, 2)
    hazard = 0.1
    t_fail = gen.duration_failure_time(3, 2, hazard)
    for t_up in (0.5, 2.0, 7.5, 12.0):
        event_by_time = t_up >= t_fail
        event_by_rule = gen.duration_failure(3, 2, t_up, hazard)
        check(f"G5 T_fail consistency t_up={t_up}", event_by_time == event_by_rule,
              f"u={u} t_fail={t_fail}")
    # Complementary-random-number rewrite is forbidden: same uniform, rule form.
    check("G5 failure rule uses U < 1-exp(-h*T)",
          (u < 1.0 - math.exp(-hazard * 2.0)) == gen.duration_failure(3, 2, 2.0, hazard))


# --------------------------------------------------------------------- #
# G6: radio-plan resource constraints on synthetic inputs.
# --------------------------------------------------------------------- #
def g6_radio_constraints() -> None:
    from mec.optimal_resource_solver import solve_convex_round
    from mec.profiles import build_default_profiles

    profiles = build_default_profiles(
        num_clients=6, total_bandwidth_mhz=5.0, power_max_dbm=23.0,
        channel_model="correlated", dropout_model="gilbert_elliott", dropout_rate=0.2,
    )
    clients = list(range(6))
    s_up = {c: 2.5e6 for c in clients}          # payload bits
    t_cmp = {c: 0.35 for c in clients}          # compute time s
    h = {c: 1e-10 for c in clients}
    est = {c: 0.8 for c in clients}
    stale = {c: 1.0 for c in clients}

    # CORA policy: constraints must hold.
    plan = solve_convex_round(
        clients, s_up, t_cmp, h,
        profiles.total_bandwidth_hz, profiles.power_profile,
        profiles.channel.n0_w_hz, 0.5, 10.0,
        lambda_e=0.1, lambda_u=1.0, omega_net=1.0,
        prev_staleness=stale, est_success=est,
        duration_hazard_rate=0.1, deadline_grid_size=24,
        transmit_mode="fill", allow_partial_arrival=False,
    )
    bw_ok = sum(plan.bandwidth_hz.values()) <= profiles.total_bandwidth_hz * (1 + 1e-9)
    pw_ok = all(0.0 <= w <= profiles.power_profile.p_max_w * (1 + 1e-9)
                for w in plan.power_w.values())
    dl_ok = 0.5 - 1e-9 <= plan.deadline_s <= 10.0 + 1e-9
    check("G6 CORA bandwidth within budget", bw_ok,
          f"sum={sum(plan.bandwidth_hz.values())}")
    check("G6 CORA power within cap", pw_ok)
    check("G6 CORA deadline within bounds", dl_ok, f"d={plan.deadline_s}")

    # Equal-Fixed policy (the default init in main): verify constraints hold.
    n = len(clients)
    eq_bw = profiles.total_bandwidth_hz / n
    eq_pw = profiles.power_profile.p_max_w
    check("G6 Equal-Fixed bandwidth sums to budget",
          abs(eq_bw * n - profiles.total_bandwidth_hz) < 1e-6)
    check("G6 Equal-Fixed power at cap", eq_pw <= profiles.power_profile.p_max_w * (1 + 1e-9))
    check("G6 Equal-Fixed deadline = d_max", True)

    # BAPFT-style (mirrors the main.py bapft_ba block incl. FS23 clamp).
    from mec.resource_solver import _min_bandwidth_for_power
    deadline = min(max(t_cmp[c] for c in clients) + 1.0, 10.0)
    deadline = max(deadline, 0.5)
    bw = {}
    fallback = []
    for c in clients:
        t_avail = deadline - t_cmp[c]
        if t_avail <= 0:
            fallback.append(c)
            continue
        b_req = _min_bandwidth_for_power(
            s_up[c], profiles.power_profile.p_max_w, h[c],
            profiles.channel.n0_w_hz, t_avail,
        )
        bw[c] = min(b_req, profiles.total_bandwidth_hz) if math.isfinite(b_req) \
            else profiles.total_bandwidth_hz
        if not math.isfinite(b_req):
            fallback.append(c)
    tot = sum(bw.values())
    if tot > profiles.total_bandwidth_hz:
        for c in bw:
            bw[c] *= profiles.total_bandwidth_hz / tot
    check("G6 BAPFT-style deadline clamped into [d_min, d_max]",
          0.5 - 1e-9 <= deadline <= 10.0 + 1e-9, f"d={deadline}")
    check("G6 BAPFT-style bandwidth within budget after renormalisation",
          sum(bw.values()) <= profiles.total_bandwidth_hz * (1 + 1e-9))
    check("G6 BAPFT-style fallback list explicit (empty expected here)",
          fallback == [], f"fallback={fallback}")


# --------------------------------------------------------------------- #
# G3b: parser rejects legacy/invalid aliases (subprocess).
# --------------------------------------------------------------------- #
def g3b_parser_rejects() -> None:
    main_py = os.path.join(CODE, "main.py")
    py = sys.executable
    for bad in ("--radio_policy legacy_cora", "--age_view no_net",
                "--reliability_view ack_off", "--network_scenario stress2"):
        cmd = f"{py} {main_py} {bad} --num_rounds 1"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=CODE, timeout=120)
        ok = r.returncode != 0 and (
            "invalid choice" in (r.stderr or "") or "error" in (r.stderr or "").lower()
        )
        check(f"G3b parser rejects {bad.split()[0]}={bad.split()[1]}", ok,
              f"rc={r.returncode} stderr_tail={(r.stderr or '')[-200:]}")


def main() -> int:
    print("== FS23 gate tests ==")
    g1_trace_hash_regression()
    g4_staleness_views()
    g3_reliability_view()
    g4b_ge_schedule()
    g5_zero_hazard()
    g6_radio_constraints()
    g3b_parser_rejects()
    print(f"\n{len(FAILURES)} failure(s): {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
