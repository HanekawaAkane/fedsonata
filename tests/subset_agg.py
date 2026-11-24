import csv
import json
import math
import os
import sys

BASE = '/data/fedsonata_journal/fs23_experiments_20261117/results/'


def load(var, ds, seed):
    d = os.path.join(BASE, 'subset_%s_%s_s%s' % (var, ds, seed))
    if not os.path.exists(os.path.join(d, 'DONE')):
        return None
    fin = json.load(open(os.path.join(d, 'final_metrics.json')))
    rounds = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl')) if l.strip()]
    rounds = [r for r in rounds if r.get('row_type') == 'round_summary']
    cum = {
        'ul_mb': sum(r.get('upload_bytes', 0) for r in rounds) / 1e6,
        'tx_e': sum(r.get('tx_energy_j', 0) for r in rounds),
        'mec_s': sum(r.get('mec_latency_s', 0) for r in rounds),
        'lpc': sum(r.get('lpc_events', 0) for r in rounds),
        'on_time': (sum(r.get('on_time_rate', 0) for r in rounds) / max(1, len(rounds))),
    }
    return {
        'acc': (fin.get('accuracy_final_round') or 0) * 100,
        'acc_sel': (fin.get('accuracy_validation_selected') or 0) * 100,
        **cum,
    }


def agg(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def main():
    variants = ['baseline', 'I1', 'I2', 'I3']
    datasets = ['cifar100', 'cifar10']
    seeds = ['42', '2025', '2026']
    table = {}
    for var in variants:
        for ds in datasets:
            rows = [load(var, ds, s) for s in seeds]
            rows = [r for r in rows if r]
            if not rows:
                continue
            stats = {}
            for k in ('acc', 'acc_sel', 'ul_mb', 'tx_e', 'mec_s', 'lpc', 'on_time'):
                m, sd = agg([r[k] for r in rows])
                stats[k] = (m, sd)
            table[(var, ds)] = (stats, len(rows))

    print('== subset summary (mean +/- sd over seeds) ==')
    hdr = '%-22s %-9s %10s %10s %9s %8s %8s %7s' % ('variant', 'dataset', 'acc', 'UL_MB', 'tx_E', 'MEC_s', 'lpc', 'on_time')
    print(hdr)
    for (var, ds), (st, n) in sorted(table.items()):
        print('%-22s %-9s %6.2f+-%.2f %6.1f+-%.1f %6.1f+-%.1f %5.2f+-%.2f %5.0f+-%.0f %6.3f+-%.3f' % (
            var, ds,
            st['acc'][0], st['acc'][1], st['ul_mb'][0], st['ul_mb'][1], st['tx_e'][0], st['tx_e'][1],
            st['mec_s'][0], st['mec_s'][1], st['lpc'][0], st['lpc'][1], st['on_time'][0], st['on_time'][1]))

    # paired per-seed diffs vs baseline
    print('\n== paired diff vs baseline (idea - baseline) ==')
    for var in ('I1', 'I2', 'I3'):
        for ds in datasets:
            diffs = {}
            for s in seeds:
                a, b = load(var, ds, s), load('baseline', ds, s)
                if not a or not b:
                    continue
                for k in ('acc', 'ul_mb', 'tx_e', 'lpc'):
                    diffs.setdefault(k, []).append(a[k] - b[k])
            if not diffs:
                continue
            parts = []
            for k, v in diffs.items():
                m, sd = agg(v)
                parts.append('%s=%+.2f+-%.2f' % (k, m, sd))
            print('%-4s %-9s n=%d  %s' % (var, ds, len(diffs.get('acc', [])), '  '.join(parts)))

    # csv export
    out = os.path.join(BASE, 'fs23_aggregate', 'subset_screen.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['variant', 'dataset', 'seed', 'acc_final', 'acc_sel', 'ul_mb', 'tx_e', 'mec_s', 'lpc', 'on_time'])
        for var in variants:
            for ds in datasets:
                for s in seeds:
                    r = load(var, ds, s)
                    if r:
                        w.writerow([var, ds, s] + ['%.4f' % r[k] for k in ('acc', 'acc_sel', 'ul_mb', 'tx_e', 'mec_s', 'lpc', 'on_time')])
    print('\nwritten', out)


if __name__ == '__main__':
    main()
