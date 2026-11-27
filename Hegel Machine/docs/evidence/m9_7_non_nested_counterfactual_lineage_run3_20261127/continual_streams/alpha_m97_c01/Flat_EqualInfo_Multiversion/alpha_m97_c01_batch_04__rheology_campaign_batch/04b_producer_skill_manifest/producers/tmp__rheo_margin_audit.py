exec(open('/tmp/rheo_cv_final.py').read().split('for lab in (*np.unique(labs), -1):')[0])
import json
from pathlib import Path
from types import ModuleType
src = (p/'alpha_m97_c01_batch_04_solver.py').read_text()
sweeps = json.loads(s._ARCHIVE.read_text())[-3:]
def archive_fit(sweeps, lab):
    blocks = []
    for sw in sweeps:
        ix = np.asarray(sw['lab_id']) == lab
        u, v = (np.asarray(sw[k])[ix] for k in ('cmd_stress','true_stress'))
        keep = ~s._spikes(u,v)
        blocks.append(np.stack((u[keep],v[keep])))
    return s._line(*np.concatenate(blocks,axis=1))
print('Prefix sweep forecast, maximum physical stress error on clean held-out readings: latest / pooled')
for j in (1,2):
    err = [[],[]]
    sw = sweeps[j]
    for lab in range(4):
        ix = np.asarray(sw['lab_id']) == lab
        u,v = (np.asarray(sw[k])[ix] for k in ('cmd_stress','true_stress'))
        keep = ~s._spikes(u,v)
        for k, prefix in enumerate((sweeps[j-1:j],sweeps[:j])):
            err[k].extend(np.abs(np.polyval(archive_fit(prefix,lab),u[keep])-v[keep]))
    print(sw['episode_short_name'],np.max(err,axis=1))
print('Current channel 3 two-point calibration error from the latest archived line:', np.abs(np.polyval(archive_fit(sweeps[-1:],3),d['calibration_load_cells']['cmd_stress'])-d['calibration_load_cells']['true_stress']))
for cutoff in ('.050','.075','.110'):
    m=ModuleType('probe'); exec(src.replace('profile(u, .075)','profile(u, '+cutoff+')').replace('profile(final.x, .075)','profile(final.x, '+cutoff+')'),m.__dict__)
    metrics=[]; taus=[]
    for lab in np.unique(labs):
        f=labs!=lab; g=labs==lab
        tau,w=m._fit(dt,t,*(a[f] for a in args),yc[f],yr[f],~missing[f])
        c,r=m._design(tau,dt,t,*(a[g] for a in args))
        e=np.abs(c@w-yc[g]); z=np.abs(r@w-yr[g]); complete=~missing[g]
        metrics.append([np.max(np.median(e[complete],axis=0)),np.median(z[complete]),np.median(e[~complete]),np.median(z[~complete])]); taus.append(tau)
    print('cutoff',cutoff,'worst complete horizon strain / complete rate / imputed strain / imputed rate medians',np.max(metrics,axis=0).round(6),'tau fold range',np.ptp(taus,axis=0).round(6))
