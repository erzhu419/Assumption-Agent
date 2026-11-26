import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_w04_batch_06_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w04_batch_06.pkl','rb'))[0]['dataset']
tr,te=d['train'],d['test']; dt,t,q=(d[k] for k in ('past_dt','eval_times','yield_stress'))
g=m._calibrate(d['calibration_load_cells']); xa,xb=m._align(tr,te)
truth_tau=np.array([.53,3.6]); truth_w=np.array([.67,.58,.081,.149,-.129,.109,-.089,.043,.031])
def reference(hist,hold,x,taus,w,yield_stress):
    state=np.zeros((len(hold),2)); work=np.zeros(len(hold))
    for j,step in enumerate(dt):
        decay=np.exp(-step/taus)
        state=state*decay+hist[:,j,None]*(1-decay)
        work+=step*np.maximum(abs(hist[:,j])-yield_stress,0)**2
    drift=w[2]+x@w[3:7]+w[7]*np.sign(hold)*np.maximum(abs(hold)-yield_stress,0)**2
    amplitude=(hold[:,None]-state)*w[:2]
    amplitude[:,0]+=w[8]*work*hold
    return amplitude@(1-np.exp(-t[None,:]/taus[:,None]))+drift[:,None]*t, amplitude@(1/taus)+drift
for scale in (1.,2.):
    synth={**d}
    for k,s,x in [('train',tr,xa),('test',te,xb)]:
        sc={**s}; gain,bias=g[s['lab_id']].T
        h=scale*(s['past_stress_cmd']*gain[:,None]+bias[:,None]); u=scale*(s['stress_hold_cmd']*gain+bias)
        sc['past_stress_cmd']=(h-bias[:,None])/gain[:,None]; sc['stress_hold_cmd']=(u-bias)/gain
        y,r=reference(h,u,x,truth_tau,truth_w,q); sc['instant_rate_obs']=r
        if k=='train': sc['strain_curves_obs']=s['strain_0'][:,None]+y
        else: expected=(s['strain_0'][:,None]+y,r)
        synth[k]=sc
        print('stress scale',scale,k,'history/positive hold/negative hold above yield:',int(np.any(abs(h)>q,axis=1).sum()),int((u>q).sum()),int((u< -q).sum()))
    out=m.solve_alpha_w04_batch_06(synth)
    truths=(truth_tau,truth_w[:7],truth_w[7:] if scale>1 else np.zeros(2),*expected)
    errors={k:float(np.max(abs(out[k]-a))) for k,a in zip(('estimated_taus','hereditary_weights','viscoplastic_weights','predicted_strain','initial_rate_pred'),truths)}
    print('noiseless recovery max errors',errors)
    assert max(errors.values())<3e-5
    if scale==1: assert np.array_equal(out['viscoplastic_weights'],np.zeros(2))
# Sweep plus an EMI glitch, and no checks in the opposite wing. The fit
# should reject the glitch and still select the historical shifted regime.
cmd=np.array([-1.8,-.8,0.,.7,1.4,2.1]); measured=g[0,0]*cmd+g[0,1]; measured[2]+=.35
cal=m._calibrate(dict(lab_id=np.zeros(len(cmd),int),cmd_stress=cmd,true_stress=measured))
print('EMI case max affine error',np.max(abs(g-cal)))
assert np.max(abs(g-cal))<1e-9
