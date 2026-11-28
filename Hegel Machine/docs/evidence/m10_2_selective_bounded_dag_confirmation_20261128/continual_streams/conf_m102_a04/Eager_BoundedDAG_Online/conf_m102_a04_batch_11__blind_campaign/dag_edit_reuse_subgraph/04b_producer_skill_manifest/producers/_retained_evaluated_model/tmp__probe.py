import pickle, runpy, glob
import numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); lin=a['lin']; one=a['one']; p=s['panel_a']; U=p['u']; V=p['v']; mask=np.max(abs(V),axis=1)<4; u=U[mask]; v=V[mask]; x=s['train']['x']; y=np.column_stack([s['train']['y'],s['train']['traj']]); c=s['panel_c'];
def score(z, split=52):
 w=a['_panel_maps'](c,z); w=(w-w[:split].mean(0))/np.maximum(w[:split].std(0),.1); g=w@w.T/w.shape[1]; d=g.diagonal(); dist=np.maximum(d[:,None]+d-2*g,0); q=(1+.25*g)**2; m=np.arange(len(x))<split; p=np.mean([k[~m][:,m]@np.linalg.solve(k[m][:,m]+r*np.eye(m.sum()),y[m]) for k,r in [(q,3e-5),(q+np.exp(-.06*dist),3e-4)]],0); e=abs(p-y[~m]).mean(0); return np.round([e[0],e[1:].mean()],6)
def fit(name,f,init=None):
 if init is not None:
  q=least_squares(lambda q:(f(u,q)@lin(f(u,q),v)-v).ravel(),init,max_nfev=200).x; F=lambda z:f(z,q)
 else:q=[]; F=f
 A=F(u); C=lin(A,v); r=A@C-v; h=np.sum(np.linalg.qr(A)[0]**2,axis=1); rm=lambda e:np.sqrt(np.mean(e**2)); print(name,'q',np.round(q,5),'rms',rm(r),'loo',rm(r/np.maximum(1-h[:,None],1e-8)),'max',abs(r).max());
 if name in ('poly2','add3','poly3','affine','trig','tanh','rational','soft','add23'): print('co',np.round(C,4).T)
 base=lin(one(u),v); z0=(x-base[0])@np.linalg.inv(base[1:]); sol=least_squares(lambda z:(F(z.reshape(-1,4))@C-x).ravel(),z0.ravel(),max_nfev=35).x.reshape(-1,4); print('  CV',score(sol),'Fwd RMS',rm(F(sol)@C-x),'zmax',abs(sol).max()); return F,C
if __name__=='__main__':
 print('row mask',np.where(~mask)[0]); print('raw',score(x)); o,t=a['_align'](p); print('diagonal',score((x-o)/t));
 fit('affine',one)
 for d in (2,3):fit('poly'+str(d),lambda z:np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(4),k)]))
 fit('add3',lambda z:np.c_[one(z),z**3]); fit('add23',lambda z:np.c_[one(z),z**2,z**3]);
 for name,g in [('trig',np.sin),('tanh',np.tanh),('rational',lambda z:z/(1+z*z)),('soft',lambda z:z/(1+abs(z)))]:fit(name,lambda z,q:np.c_[one(z),g(z*q[0])],[1.4]);
