import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w01_batch_04_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_w01_batch_04.pkl','rb'))[0]['dataset']
tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; sy=d['yield_stress']; c=d['calibration_load_cells']
x,_,complete,_=s._align(tr,te); maps=s._calibrate(c)
h,hold=s._stress(tr,maps); ov,w=s._plastic(h,hold,dt,sy)
for j in range(4):
    b=tr['lab_id']==j; a=~b
    tau,theta=s._fit({k:v[a] for k,v in tr.items()},x[a],complete[a],maps,dt,t,sy)
    A,B=s._design(tau,h[b],hold[b],x[b],dt,t,ov[b],w[b]); r=B@theta
    e=tr['strain_0'][b,None]+A@theta-tr['strain_curves_obs'][b]; re=r-tr['instant_rate_obs'][b]
    masks=[('all',np.ones(b.sum(),bool)),('complete',complete[b]),('missing',~complete[b]),('below',(ov[b]==0)&(w[b]==0)),('above',(ov[b]!=0)|(w[b]!=0))]
    print('lab',j)
    for name,m in masks:
        if m.any():
            err=e[m]-np.where(complete[b][m],0,re[m])[:,None]*t
            print(name,int(m.sum()),'curve abs p50/90/95/99',np.round(np.quantile(abs(err),[.5,.9,.95,.99]),5),'rate abs p50/90/95',np.round(np.quantile(abs(re[m]),[.5,.9,.95]),5))
    print('signed median by horizon',np.round(np.median(e[complete[b]],axis=0),5))
for name, labs, retain, glitch in [('sparse A',(0,1),1,0),('sparse B',(2,3),1,0),('EMI A',(0,1),10,3),('EMI B',(2,3),10,3),('sparse+EMI B',(2,3),2,1)]:
    keep=np.ones(len(c['lab_id']),bool); y=c['true_stress'].copy()
    for j in labs:
        i=np.flatnonzero(c['lab_id']==j); keep[i[retain:]]=False; y[i[:glitch]]+=np.linspace(.3,.9,glitch)
    cand=s._calibrate(dict(lab_id=c['lab_id'][keep],cmd_stress=c['cmd_stress'][keep],true_stress=y[keep]))
    out=s.solve_alpha_w01_batch_04(dict(d,calibration_load_cells=dict(lab_id=c['lab_id'][keep],cmd_stress=c['cmd_stress'][keep],true_stress=y[keep])))
    print(name,'max calibration shift',np.max(abs(cand-maps)),'taus',out['estimated_taus'].round(6),'vp',out['viscoplastic_weights'].round(6))
    assert np.max(abs(cand-maps)) < .001
    assert np.isfinite(out['predicted_strain']).all()
sub=dict(d,yield_stress=1e6)
out=s.solve_alpha_w01_batch_04(sub)
assert np.array_equal(out['viscoplastic_weights'],np.zeros(2))
# Isolate the two threshold mechanisms; check their independent onset derivatives.
h2=np.array([[0.]*len(dt),[2*sy]*len(dt),[0.]*len(dt)])
hold2=np.array([2*sy,.5*sy,-2*sy]); o2,w2=s._plastic(h2,hold2,dt,sy)
assert np.array_equal(o2!=0,[True,False,True]) and np.array_equal(w2!=0,[False,True,False]) and o2[-1]<0
T=np.array([.5,4.2]); eps=1e-7
A,B=s._design(T,h2,hold2,np.ones((3,5)),dt,np.array([eps]),o2,w2)
np.testing.assert_allclose(A[:,0,7:]/eps,B[:,7:],rtol=1e-6,atol=1e-10)
print('plastic zero-activation, sign, and onset-derivative checks passed')
