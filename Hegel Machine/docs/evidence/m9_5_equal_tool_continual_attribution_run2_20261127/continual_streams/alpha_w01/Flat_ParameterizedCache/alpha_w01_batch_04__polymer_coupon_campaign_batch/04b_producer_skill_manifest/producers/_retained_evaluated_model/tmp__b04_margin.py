import runpy
import numpy as np
m = runpy.run_path('/tmp/b04_eval.py')
s,d,g,o = (m[k] for k in ('s','d','g','o'))
tr,te = d['train'],d['test']
dt,t,sy = (d[k] for k in ('past_dt','eval_times','yield_stress'))
maps = s._calibrate(d['calibration_load_cells'])
x,xt,complete,ct = s._align(tr,te)

def rmse(a):
    return np.sqrt(np.mean(a*a))

def audit(block, x, complete, tau, theta, target, tag):
    h,hold = s._stress(block,maps)
    ov,work = s._plastic(h,hold,dt,sy)
    A,B = s._design(tau,h,hold,x,dt,t,ov,work)
    E = block['strain_0'][:,None]+A@theta-target
    rate_error = B@theta-block['instant_rate_obs']
    E -= np.where(complete,0,rate_error)[:,None]*t
    for name,b in [('below',(ov==0)&(work==0)),('above',(ov!=0)|(work!=0))]:
        if b.any():
            tail = E[b,-1]
            print(tag,name,int(b.sum()),'tail signed median',round(float(np.median(tail)),6),'tail median abs',round(float(np.median(abs(tail))),6),'curve rmse',round(float(rmse(E[b])),6),'first/last median abs',np.round(np.median(abs(E[b][:,[0,-1]]),axis=0),6))
    if (~complete).any():
        for w in (0,.5,1):
            e = E[~complete]+(1-w)*rate_error[~complete,None]*t
            print(tag,'missing drift fraction',w,'curve median abs',round(float(np.median(abs(e))),6),'last median abs',round(float(np.median(abs(e[:,-1]))),6))
    return A,B

for j in range(4):
    b=tr['lab_id']==j
    tau,theta=s._fit({k:v[~b] for k,v in tr.items()},x[~b],complete[~b],maps,dt,t,sy)
    audit({k:v[b] for k,v in tr.items()},x[b],complete[b],tau,theta,tr['strain_curves_obs'][b],f'OOF lab {j}')
tau=o['estimated_taus']; theta=np.r_[o['hereditary_weights'],o['viscoplastic_weights']]
A,B=audit(te,xt,ct,tau,theta,g['te_clean'],'public')
# At t=0 both memories have zero displacement. Check every rate column on the measured stress histories.
h,hold=s._stress(te,maps)
args=(h,hold,xt,dt)
ov,work=s._plastic(h,hold,dt,sy)
eps=1e-6
Z,_=s._design(tau,*args,np.array([0.,eps,eps/2]),ov,work)
np.testing.assert_array_equal(Z[:,0],np.zeros_like(B))
np.testing.assert_allclose((4*Z[:,2]-Z[:,1])/eps,B,rtol=1e-8,atol=1e-9)
print('all nine basis columns: zero displacement and independent onset derivative passed')
print('calibration max gain/bias abs error',np.max(abs(maps-np.column_stack((g['stress_gains'],g['stress_biases']))),axis=0))
print('public output max curve/rate abs error',np.max(abs(o['predicted_strain']-g['te_clean'])),np.max(abs(o['initial_rate_pred']-g['te_true_rate0'])))
