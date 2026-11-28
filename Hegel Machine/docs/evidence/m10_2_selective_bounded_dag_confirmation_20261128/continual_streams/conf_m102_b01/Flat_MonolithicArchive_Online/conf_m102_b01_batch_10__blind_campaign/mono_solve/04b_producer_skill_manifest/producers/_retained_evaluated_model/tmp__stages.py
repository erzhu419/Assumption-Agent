import numpy as np,pickle,itertools as it
from scipy.optimize import root,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A,B,C=[s['panel_'+k] for k in 'abc']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; LS=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(u,d):return np.column_stack([np.prod(u[:,k],axis=1) for i in range(1,d+1) for k in it.combinations_with_replacement(range(u.shape[1]),i)])
def dead(u,p=(.35,1.5)):return np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1]
def cp(u):return np.column_stack([u[:,i]*u[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])
def ab(u):return np.stack([np.ones_like(u),u,np.tanh(1.8*u),np.sin(2.4*np.roll(u,2,axis=-1))],-1)
m=np.linalg.norm(A['v'],axis=1)<7; d=ab(A['u']); ka=np.array([LS(d[m,j],A['v'][m,j]) for j in range(4)])
def inv(x):return np.array([root(lambda u:(ab(u)*ka).sum(-1)-r,r,tol=1e-10).x for r in x])
kb=LS(aff(dead(B['u'])),B['v']); kc=LS(aff(cp(C['u'])),C['v1']); k2=LS(aff(poly(C['u'],2)),C['v2'])
def b(z):return aff(dead(z))@kb
def c(w):return aff(cp(w))@kc
def c2(w):return aff(poly(w,2))@k2
z=inv(x);w=b(z);q=c(w);r=c2(w)
def err(d):
 pred=d[52:]@LS(d[:52],Y[:52]); e=np.abs(pred-Y[52:]); E=np.sqrt(np.mean((d[:52]@LS(d[:52],Y[:52])-Y[:52])**2,axis=0));return np.round([e[:,0].mean(),e[:,1:].mean(),E[0],E[1:].mean()],4)
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True);print('t',t,'k2',k2,'w/q/r range',np.max(abs(w),axis=0),np.max(abs(q),axis=0),np.max(abs(r),axis=0));
 for k,h in [('z',z),('w',w),('q',q),('r',r),('qc',np.c_[q,r]),('bh',dead(z)),('cross',cp(w)),('zq',np.c_[z,q]),('qboth',np.c_[q,c(w*1.25),c(w/1.25)])]:
  print(k,err(aff(h)),err(aff(poly(h,2))) if h.shape[1]<7 else '')
 for k,d in [('w',w),('r',r),('z',z),('q',q)]:
  for fun in [lambda d:d,lambda d:np.tanh(2.15*d),lambda d:np.arcsinh(2.15*d),lambda d:d/(1+.22*d*d),lambda d:d/(1+1.25*d*d)]:
   h=fun(d);print(k,err(aff(h)),err(aff(np.c_[h,q])),err(aff(poly(h,2))))
 print('readout coefficients',LS(aff(q[:52]),Y[:52])); print('temporal error with exp modes');
 for a,b0 in [(2.15,1.25),(2.15,.22),(1.25,.22),(2.15,1.25)]:
  for d in [np.c_[-np.expm1(-a*t),-np.expm1(-b0*t)],np.c_[-np.expm1(-a*t),t*np.exp(-b0*t)],np.c_[-np.expm1(-a*t),np.sin(b0*t)*np.exp(-.22*t)]]:
   print(a,b0,np.sqrt(np.mean((d@LS(d,Y[:,1:].T)-Y[:,1:].T)**2)))
