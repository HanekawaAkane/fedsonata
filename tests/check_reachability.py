import json

def rn(d, n):
    rows = [json.loads(l) for l in open(d + '/metrics.jsonl') if l.strip()]
    return [r for r in rows if r.get('round') == n and r.get('row_type') == 'round_summary'][0]

base = '/data/fedsonata_journal/fs23_experiments_20261117/results/'
for n in (3, 8, 15):
    a = rn(base + 'fs23_20261117_base_full_s42', n)
    b = rn(base + 'fs23_20261117_base_age_off_s42', n)
    same_bw = a['bandwidth_hz'] == b['bandwidth_hz']
    same_dl = abs(a['deadline_s'] - b['deadline_s']) < 1e-12
    same_tx = abs(a['tx_energy_j'] - b['tx_energy_j']) < 1e-12
    print('round %d: bw_identical=%s deadline_identical=%s txE_identical=%s' % (n, same_bw, same_dl, same_tx))
    print('   full: dl=%.3f txE=%.4f | ageoff: dl=%.3f txE=%.4f' % (
        a['deadline_s'], a['tx_energy_j'], b['deadline_s'], b['tx_energy_j']))
    if not same_bw:
        diff = [(k, a['bandwidth_hz'][k], b['bandwidth_hz'][k]) for k in a['bandwidth_hz']
                if abs(a['bandwidth_hz'][k] - b['bandwidth_hz'][k]) > 1e-6]
        print('   differing clients: %d e.g. %s' % (len(diff), diff[:2]))

a = rn(base + 'fs23_20261117_base_full_s42', 3)
c = rn(base + 'fs23_20261117_base_no_reliability_s42', 3)
print('round3 upload bytes full=%d no_rel=%d equal=%s' % (
    a['upload_bytes'], c['upload_bytes'], a['upload_bytes'] == c['upload_bytes']))

# P1 divergence check: allocation maps in control_trace round 5
def alloc(d, n):
    for line in open(d + '/control_trace.jsonl'):
        row = json.loads(line)
        if row['round'] == n:
            return row['allocation']
a5 = alloc(base + 'fs23_20261117_base_full_s42', 5)
b5 = alloc(base + 'fs23_20261117_base_age_off_s42', 5)
same_alloc = a5 == b5
print('round5 P1 allocation full==age_off:', same_alloc)
if not same_alloc:
    ndiff = sum(1 for k in a5 if a5[k] != b5.get(k))
    print('   clients with different block sets:', ndiff, '/ 20')
