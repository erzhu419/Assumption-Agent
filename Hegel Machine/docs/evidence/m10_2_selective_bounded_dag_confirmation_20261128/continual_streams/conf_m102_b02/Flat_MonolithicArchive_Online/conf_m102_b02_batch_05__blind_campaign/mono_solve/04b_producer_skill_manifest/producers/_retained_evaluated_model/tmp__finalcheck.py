import pickle, sys, numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._calibrate(s['panel_a'],tr['x']); pa=s['panel_a']; u=pa['u'];v=pa['v']; I=m._intercept; ls=m._ls
keep=np.argsort(np.linalg.norm(I(u)@ls(I(u),v)-v,axis=1))[:-3]; u=u[keep];v=v[keep]; r=np.roll(u,-1,axis=1); df=[np.c_[np.ones(len(u)),u[:,j],r[:,j],u[:,j]*r[:,j],u[:,j]**3] for j in range(4)]; a=np.column_stack([I(u)@(ls(I(u),v))[:,j]-v[:,j] for j in range(4)]); e=np.column_stack([d@ls(d,v[:,j])-v[:,j] for j,d in enumerate(df)]); print('panel_a trimmed RMS affine/cubic',np.mean(a*a)**.5,np.mean(e*e)**.5,'cubic LOO',np.mean(np.column_stack([e[:,j]/(1-np.diag(d@np.linalg.pinv(d))) for j,d in enumerate(df)])**2)**.5,'inverse reference max',np.max(abs(m._calibrate(pa,v)-u)))
idx=np.arange(len(z)); rng=np.random.default_rng(173); groups=[idx[52:],*np.array_split(rng.permutation(len(z)),4)]; groups.extend(np.argsort(f)[-16:] for f in [np.linalg.norm(z,axis=1),np.max(abs(z),axis=1),*[z[:,j]*sg for j in range(4) for sg in (-1,1)]])
def score(p,k):
 ey=np.abs(p['y_hat']-tr['y'][k]); et=np.abs(p['state_pred']-tr['traj'][k]); return [ey.mean(),et.mean(),et.mean(0).max(),et[:,-1].mean()]
def batch(k):
 j=np.setdiff1d(idx,k); return {**s,'train':{**tr,**{h:tr[h][j] for h in ('x','y','traj')}},'test':{'x':tr['x'][k]}}
def test(fn):
 rr=np.array([score(fn(batch(k)),k) for k in groups]); return np.round(rr[0],6),np.round(rr[1:5].max(0),6),np.round(rr[5:].max(0),6)
print('artifact fixed/random worst/input-tail worst [target, trajectory, max time, endpoint]',test(m.solve_batch)); assert np.max(test(m.solve_batch))<.038
p=m.solve_batch(s); h=p['y_hat']; t=p['state_pred']; assert h.shape==(36,) and t.shape==(36,8) and all(np.isfinite(x).all() for x in p.values()); np.testing.assert_allclose(p['diagnostics_vec'],[h.mean(),h.std(),t[:,-1].mean(),t[:,-1].std()]); orig=m._b_operator
pb=s['panel_b'];u=pb['u'];v=pb['v'];
def shared(panel,x):
 def design(a,q):return I(np.c_[a,np.sin(a*q[0])]);
 def obj(q): d=design(u,q);return (d@ls(d,v)-v).ravel()
 q=least_squares(obj,[2.9],bounds=(1.5,4.5)).x; return design(x,q)@ls(design(u,q),v)
m._b_operator=shared; print('shared challenger',test(m.solve_batch));m._b_operator=orig
for name,f,init in [('sin',lambda a,q:a-q[0]*np.sin(q[1]*a),[.4,2.9]),('erf',lambda a,q:a-q[0]*erf(q[1]*a),[.4,1.45]),('tanh',lambda a,q:a-q[0]*np.tanh(q[1]*a),[.4,1.45]),('rat',lambda a,q:a-q[0]*a/(1+q[1]*a*a),[1,1]),('saturate',lambda a,q:a*np.tanh(q[0]*a*a),[1]),('exp',lambda a,q:a*(1-q[0]*np.exp(-q[1]*a*a)),[1,1]),('softclip',lambda a,q:a-np.clip(a,-q[0],q[0]),[.4])]:
 def D(a,q):return I(f(a,q));
 def obj(q):d=D(u,q);return (d@ls(d,v)-v).ravel()
 q=least_squares(obj,init,bounds=(.005,8.)).x; d=D(u,q);c=ls(d,v);r=d@c-v;loo=np.mean((r/(1-np.diag(d@np.linalg.pinv(d)))[:,None])**2)**.5; m._b_operator=lambda panel,x:D(x,q)@c;print(name,q,'panel rms/loo',np.mean(r*r)**.5,loo,'cv',test(m.solve_batch));
