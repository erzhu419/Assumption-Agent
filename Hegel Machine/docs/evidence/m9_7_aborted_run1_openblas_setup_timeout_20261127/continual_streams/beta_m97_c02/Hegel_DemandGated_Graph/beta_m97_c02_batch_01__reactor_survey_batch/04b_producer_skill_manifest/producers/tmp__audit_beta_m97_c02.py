import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate_audit', root/'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    items = pickle.load(f)

def err(a,b):
    return float(np.max(np.abs(a-b)))

def axis(k, v, robin, length):
    n = len(k)
    h = length/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
            a[i,i] = k[i]/h**2 + robin/h
        else:
            a[i,i-1] = -(k[i-1]+k[i])/(2*h*h) - max(v[i],0)/h
            a[i,i+1] = -(k[i+1]+k[i])/(2*h*h) - max(-v[i],0)/h
            a[i,i] = -a[i].sum()
    return a

for case,item in enumerate(items):
    s,g = item['study'],item['gt']
    nx,ny = s['nx'],s['ny']
    amp = sum(max(0.,v-s['critical_vortex_circulation'])**2 for v in s['past_vortex_circulation_trace'])
    x,y = np.meshgrid(np.linspace(0.,1.,nx),np.linspace(0.,1.,ny),indexing='ij')
    sink = (0.95+0.55*np.tanh(amp))*(np.exp(-16*x*x)+np.exp(-16*(1-x)**2))*(0.6+0.4*np.sin(np.pi*y))
    i,j = np.asarray(s['sensor_ij']).T
    rhs = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
    for mode in (0,1):
        sealed = mode == 0 and amp > 0
        ax = axis(s['kappa_x'],s['vel_x'],0. if sealed else s['robin_x'],s['Lx'])
        ay = axis(s['kappa_y'],s['vel_y'],0. if sealed else s['robin_y'],s['Ly'])
        if sealed:
            assert max(np.max(np.abs(ax.sum(1))),np.max(np.abs(ay.sum(1)))) < 1e-10
        L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag((s['V_reaction_2d']+(sink if sealed else 0.)).ravel(order='F'))
        T = [np.linalg.solve(L+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
        got,terminal = m._forward(dict(s,recirculation_loop_mode=mode))
        e = max(err(got,np.concatenate([t[i+nx*j] for t in T])),err(terminal,T[-1]))
        print('case',case,'loop mode',mode,'dense forward max error',e)
        assert e < 1e-11
    sweeps=s['sensor_calibration_sweeps']
    bank,ref,raw = [np.asarray(sweeps[k]) for k in ('bank_id','ref_signal','raw_readout')]
    rowbank=np.asarray(s['row_bank_id'])
    Z=s['Z_nuisance']
    def profile(v):
        return v-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@v)
    unaligned=profile(np.asarray(s['y_obs_raw']))
    for b in range(4):
        heldout=[]; weight=[]; field=[]; adjusted=[]
        ids=np.flatnonzero(bank==b)
        for k in ids:
            train=ids[ids!=k]
            slope,offset=np.polyfit(ref[train],raw[train],1)
            heldout.append(abs(slope*ref[k]+offset-raw[k]))
            keep=np.arange(bank.size)!=k
            st=dict(s,sensor_calibration_sweeps={key:np.asarray(val)[keep] for key,val in sweeps.items()})
            out=m.solve_beta_m97_c02_batch_01(st)
            weight.append(err(out['weights'],g['w_opt']))
            field.append(err(out['terminal_field'],g['terminal_field_A']))
            adjusted.append(err(profile(m._calibration(st))[rowbank==b],g['r_y'][rowbank==b]))
        unadjusted=err(unaligned[rowbank==b],g['r_y'][rowbank==b])
        print('case',case,'bank',b,'LOO sweep',max(heldout),'profiled adjusted/unadjusted',max(adjusted),unadjusted,'weight',max(weight),'field',max(field))
        assert max(heldout)<1e-5 and max(adjusted)<1e-6 and max(adjusted)<unadjusted
        assert max(weight)<2e-6 and max(field)<2e-7
