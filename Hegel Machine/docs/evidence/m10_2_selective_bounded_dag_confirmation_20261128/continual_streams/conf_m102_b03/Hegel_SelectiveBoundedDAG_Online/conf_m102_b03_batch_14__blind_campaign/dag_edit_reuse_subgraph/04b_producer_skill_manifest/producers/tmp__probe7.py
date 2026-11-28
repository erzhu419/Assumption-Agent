import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; i,j=np.array(list(itertools.combinations(range(4),2))).T
pairs=list(itertools.combinations(range(6),4))+[tuple(range(6))]; bases=[np.ones((len(u),1)),np.c_[np.ones(len(u)),u]]
fs={'id':lambda x:x,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'soft':lambda x:x/(1+abs(x)),'rat':lambda x:x/(1+x*x),'gauss':lambda x:np.exp(-x*x),'sqrat':lambda x:x*x/(1+x*x)}
for k in (1,2):
 v=s['panel_c']['v'+str(k)]; out=[]
 for name,f in fs.items():
  for kind in ('coord','prod','preprod','left','sum','diff','prodadd'):
   for B in bases:
    def build(a):
     if kind=='coord':return f(a*u)
     if kind=='prod':return f(a*u[:,i]*u[:,j])
     if kind=='preprod':return f(a*u[:,i])*f(a*u[:,j])
     if kind=='left':return f(a*u[:,i])*u[:,j]
     if kind=='sum':return f(a*(u[:,i]+u[:,j]))
     if kind=='diff':return f(a*(u[:,i]-u[:,j]))
     return u[:,i]*u[:,j]+f(a*(u[:,i]+u[:,j]))
    for jj in ([tuple(range(4))] if kind=='coord' else pairs):
     def score(a,cv=False):
      A=np.c_[B,build(a)[:,jj]]; P=np.linalg.pinv(A); e=A@(P@v)-v; h=np.einsum('ij,ji->i',A,P); return np.mean((e/(np.maximum(1-h,1e-8)[:,None] if cv else 1))**2)**.5
     grid=[.1,.22,.45,.84,1.25,1.7,2.15,2.4,2.9,3.5,4.5,6.]; ix=np.argmin([score(a) for a in grid]); a=minimize_scalar(score,bounds=(grid[max(0,ix-1)],grid[min(ix+1,len(grid)-1)]),method='bounded').x
     out.append((score(a,True),score(a),name,kind,B.shape[1],jj,a))
 print(k,*sorted(out)[:17],sep='\n',flush=True)
