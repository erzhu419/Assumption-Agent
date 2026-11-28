import numpy as np,pickle
from itertools import combinations
from scipy.optimize import least_squares
from ops import _branches,_one,_fit,_calibration,_inverse
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; n=52
x=tr['x']; z=_inverse(x,_calibration(s['panel_a'])); c,d=_branches(s,x)
p=s['panel_b']; bmat=_fit(_one(p['u']),p['v']); b=_one(z)@bmat
C=_fit(_one(s['panel_c']['u']*np.roll(s['panel_c']['u'],-1,axis=1)),s['panel_c']['v1']); D=_fit(_one(np.cos(2.4*s['panel_c']['u'])),s['panel_c']['v2']);
def branches(z):return _one(z*np.roll(z,-1,axis=1))@C,_one(np.cos(2.4*z))@D
print('inverse max residual',np.max(abs(x-(_calibration(s['panel_a'])[:,0]+_calibration(s['panel_a'])[:,1]*z+_calibration(s['panel_a'])[:,2]*np.roll(z,-1,axis=1)+_calibration(s['panel_a'])[:,3]*z*np.roll(z,-1,axis=1)+_calibration(s['panel_a'])[:,4]*z**3))))
def modes(r):return -np.expm1(-t[:,None]*r)
def res(r):
 q=modes(r);return (q@_fit(q,T[:n].T)-T[:n].T).ravel()
r=least_squares(res,[.39,1.8,4.8]).x; q=modes(r); A=_fit(q,T.T).T; print('rates',r)
def score(name,h,verbose=False):
 h=_one(h); by=_fit(h[:n],y[:n]); bt=_fit(h[:n],T[:n]); ey=h@by-y; et=h@bt-T; print(name,h.shape[1],*[round(np.mean(abs(e[sl])),5) for sl in [slice(0,n),slice(n,None)] for e in [ey,et]],'amp',np.mean(abs((h@_fit(h[:n],A[:n])-A)[n:]),0).round(4));
 if verbose: print('y weights',by.round(3)); print('mode weights',_fit(h[:n],A[:n]).round(3))
outer=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for name,u in [('z',z),('b',b),('x',x),('invb',(z-bmat[0])@np.linalg.inv(bmat[1:]))]:
 c,d=branches(u); dot=np.sum(c*d,axis=1); h=np.c_[c,d,dot];
 for nm,f in [('base',h),('base+cross01',np.c_[h,c[:,0]*d[:,1]]),('base+u',np.c_[h,u]),('all cross',np.c_[c,d,outer(c,d)]),('squares and all cross',np.c_[c,d,outer(c,d),c*c,d*d]),('tanh',np.c_[h,np.tanh(c),np.tanh(d)]),('all quadratics',np.c_[c,d,outer(c,d),*[g[:,i]*g[:,j] for g in (c,d) for i in range(4) for j in range(i,4)]])]: score(name+' '+nm,f,nm=='base+cross01')
for name,g in [('c',branches(z)[0]),('d',branches(z)[1]),('b',b),('c+d',sum(branches(z))),('c*d',np.prod(branches(z),axis=0)),('cz',branches(z)[0]*z),('dz',branches(z)[1]*z)]:
 print(name,'correlations',np.corrcoef(np.c_[y[:n],A[:n],g[:n]].T)[:4,4:].round(2))
