import json
import math
import os

BASE = '/data/fedsonata_journal/fs23_experiments_20261117/results/'


def rounds_of(d):
    rows = [json.loads(l) for l in open(os.path.join(BASE, d, 'metrics.jsonl')) if l.strip()]
    return sorted([r for r in rows if r.get('row_type') == 'round_summary'], key=lambda r: r['round'])


def final_of(d):
    return json.load(open(os.path.join(BASE, d, 'final_metrics.json')))


def phase_stats(rows, lo, hi):
    sub = [r for r in rows if lo <= r['round'] <= hi]
    if not sub:
        return None
    n = len(sub)
    return {
        'acc': sum(r.get('accuracy_personalized') or 0 for r in sub) / n,
        'on_time': sum(r.get('on_time_rate') or 0 for r in sub) / n,
        'ul_mb': sum(r.get('upload_bytes') or 0 for r in sub) / 1e6,
        'tx_E': sum(r.get('tx_energy_j') or 0 for r in sub),
        'mec_s': sum(r.get('mec_latency_s') or 0 for r in sub),
        'dl_miss': sum(r.get('deadline_miss_count') or 0 for r in sub),
        'base_drop': sum(r.get('base_dropout_clients') or 0 for r in sub),
        'dur_drop': sum(r.get('duration_dropout_clients') or 0 for r in sub),
        'rel': sum(r.get('reliability_mean') or 0 for r in sub) / n,
        'lpc': sum(r.get('lpc_events') or 0 for r in sub),
    }


def show(tag, d):
    rows = rounds_of(d)
    fin = final_of(d)
    print('== %s (%s) ==' % (tag, d))
    print('  final_round_acc=%.3f%% val_sel_acc=%.3f%% rounds=%s' % (
        (fin.get('accuracy_final_round') or 0) * 100,
        (fin.get('accuracy_validation_selected') or 0) * 100,
        fin.get('completed_rounds')))
    for name, lo, hi in (('p1(1-10)', 1, 10), ('p2-stress(11-15)', 11, 15), ('p3-recover(16-20)', 16, 20)):
        st = phase_stats(rows, lo, hi)
        if st is None:
            continue
        print('  %-18s acc=%.3f on_time=%.3f ul=%.1fMB txE=%.1f mec=%.2fs miss=%d drop=%d/%d rel=%.3f lpc=%d' % (
            name, st['acc'] * 100, st['on_time'], st['ul_mb'], st['tx_E'], st['mec_s'],
            st['dl_miss'], st['base_drop'], st['dur_drop'], st['rel'], st['lpc']))
    return rows, fin


for seed in ('42', '2025', '2026'):
    show('IDEAL FULL s' + seed, 'fs23_20261117_ideal_full_s' + seed)
    show('IDEAL DCO s' + seed, 'fs23_20261117_ideal_delivery_control_open_s' + seed)

for seed in ('42', '2025', '2026'):
    show('SHIFT FULL s' + seed, 'fs23_20261117_shift_full_s' + seed)
    show('SHIFT DCO s' + seed, 'fs23_20261117_shift_delivery_control_open_s' + seed)

# IDEAL equivalence check: FULL vs DELIVERY_CONTROL_OPEN per-round metrics must match
print('\n== IDEAL equivalence (FULL vs DELIVERY_CONTROL_OPEN) ==')
for seed in ('42', '2025', '2026'):
    a = rounds_of('fs23_20261117_ideal_full_s' + seed)
    b = rounds_of('fs23_20261117_ideal_delivery_control_open_s' + seed)
    same = True
    diffs = []
    for ra, rb in zip(a, b):
        for k in ('accuracy_personalized', 'upload_bytes', 'tx_energy_j', 'mec_latency_s', 'on_time_rate'):
            va, vb = ra.get(k), rb.get(k)
            if va is None or vb is None:
                continue
            if abs(va - vb) > 1e-9 * max(1.0, abs(va)):
                same = False
                diffs.append((ra['round'], k, va, vb))
    fa, fb = final_of('fs23_20261117_ideal_full_s' + seed), final_of('fs23_20261117_ideal_delivery_control_open_s' + seed)
    feq = abs((fa.get('accuracy_final_round') or 0) - (fb.get('accuracy_final_round') or 0)) < 1e-12
    print('  s%s: per_round_identical=%s final_identical=%s ndiff=%d %s' % (
        seed, same, feq, len(diffs), diffs[:3]))

# IDEAL zero-miss check
print('\n== IDEAL zero deadline miss / zero dropout ==')
for seed in ('42', '2025', '2026'):
    for arm in ('full', 'delivery_control_open'):
        rows = rounds_of('fs23_20261117_ideal_%s_s%s' % (arm, seed))
        miss = sum(r.get('deadline_miss_count') or 0 for r in rows)
        drop = sum(r.get('base_dropout_clients') or 0 for r in rows) + sum(r.get('duration_dropout_clients') or 0 for r in rows)
        ontime = min(r.get('on_time_rate') or 1 for r in rows)
        print('  %s s%s: miss=%d drop=%d min_on_time=%.3f' % (arm, seed, miss, drop, ontime))

# SHIFT paired FULL - DCO per phase
print('\n== SHIFT paired FULL - DCO (per phase) ==')
for name, lo, hi in (('p1', 1, 10), ('p2-stress', 11, 15), ('p3-recover', 16, 20)):
    acc_d, ul_d, tx_d, ot_d = [], [], [], []
    for seed in ('42', '2025', '2026'):
        ra = phase_stats(rounds_of('fs23_20261117_shift_full_s' + seed), lo, hi)
        rb = phase_stats(rounds_of('fs23_20261117_shift_delivery_control_open_s' + seed), lo, hi)
        acc_d.append((ra['acc'] - rb['acc']) * 100)
        ul_d.append(ra['ul_mb'] - rb['ul_mb'])
        tx_d.append(ra['tx_E'] - rb['tx_E'])
        ot_d.append(ra['on_time'] - rb['on_time'])
    def ms(v):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        return m, sd
    for label, v in (('acc_pp', acc_d), ('ul_mb', ul_d), ('txE_J', tx_d), ('on_time', ot_d)):
        m, sd = ms(v)
        print('  %-10s %-8s mean=%+.3f sd=%.3f vals=%s' % (name, label, m, sd, [round(x, 3) for x in v]))

# SHIFT recovery gap: best pre-stress vs round-20
print('\n== SHIFT recovery gap (FULL) ==')
for seed in ('42', '2025', '2026'):
    rows = rounds_of('fs23_20261117_shift_full_s' + seed)
    pre = max(r['accuracy_personalized'] for r in rows if r['round'] <= 10)
    stress_min = min(r['accuracy_personalized'] for r in rows if 11 <= r['round'] <= 15)
    r20 = [r for r in rows if r['round'] == 20][0]['accuracy_personalized']
    print('  s%s: pre_best=%.3f stress_min=%.3f round20=%.3f gap_to_pre=%.3f recovered=%s' % (
        seed, pre * 100, stress_min * 100, r20 * 100, (pre - r20) * 100, r20 >= pre - 0.2))
