import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A,B,C=[s['panel_'+k] for k in 'abc']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; LS=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(u,d):return np.column_stack([np.prod(u[:,k],axis=1) for i in range(1,d+1) for k in it.combinations_with_replacement(range(4),i)])
def dead(u,p=(.35,1.5)):return np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1]
def cp(u):return np.column_stack([u[:,i]*u[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])
def fit(d,y):
 c=LS(d,y); return np.sqrt(np.mean((d@c-y)**2)),c
def ba(u):return aff(dead(u))@kb
def c1(u):return aff(cp(u))@kc
kb=fit(aff(dead(B['u'])),B['v'])[1];kc=fit(aff(cp(C['u'])),C['v1'])[1];print('B C1 RMS',fit(aff(dead(B['u'])),B['v'])[0],fit(aff(cp(C['u'])),C['v1'])[0]);np.set_printoptions(precision=4,suppress=True)
u=C['u'];v=C['v2']; print('C2 quad',fit(aff(poly(u,2)),v)); print('C2 quadratic residual', (aff(poly(u,2))@LS(aff(poly(u,2)),v)-v)[:6]);
u=A['u'];v=A['v']; m=np.linalg.norm(v,axis=1)<7;print('A max norms',np.sort(np.linalg.norm(v,axis=1)));u=u[m];v=v[m];
for j in range(4):
 z=u[:,j];o=u[:,(j+2)%4]; n=u[:,(j+1)%4];p=u[:,(j-1)%4];
 ds={'poly3':poly(u,3),'poly2':poly(u,2),'known':np.c_[z,np.tanh(1.8*z),np.sin(2.4*o)],'cross':np.c_[z,o,z**3,o**3,z*o,z*z*o,z*o*o], 'split':np.c_[z,np.tanh(1.8*z),o,np.sin(2.4*o),n,p,np.sin(2.4*n),np.sin(2.4*p),z*o,z*n,z*p]}
 for k,d in ds.items():
  d=aff(d);r,c=fit(d,v[:,j]);print('A',j,k,r,c if k in ('known','cross','split') else '')
z0=np.column_stack([np.interp(x[:,j],np.sort(v[:,j]),np.sort(u[:,j])) for j in range(4)]); ka=LS(aff(u),v);za=(x-ka[0])@np.linalg.inv(ka[1:]); print('z ranges',za.min(0),za.max(0));
def error(d):
 pred=d[52:]@LS(d[:52],Y[:52]); e=np.abs(pred-Y[52:]); return e[:,0].mean(),e[:,1:].mean()
for nm,z in [('x',x),('inva',za),('quantile',z0)]:
 for name,w in [(nm,z),(nm+'B',ba(z)),(nm+'C1',c1(z)),(nm+'BC1',c1(ba(z))), (nm+'BC2',aff(poly(ba(z),2))@LS(aff(poly(C['u'],2)),C['v2']))]:
  print(name,[(d,error(aff(poly(w,d)))) for d in (1,2,3)]);print(' comb z/w',error(aff(np.c_[poly(w,2),z,w*np.roll(z,-1,axis=1),w*np.roll(z,1,axis=1),w*z])))
print('readout y-vs-traj',error(aff(tr['traj'])), 'trajectory head',np.linalg.svd(tr['traj'],full_matrices=False)[2][:2])
