import numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist

def aff(u): return np.c_[np.ones(len(u)),u]
def quad(u): return np.column_stack([aff(u)]+[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(u.shape[1]),2)])
def maps(s,xx,debug=False):
 a,b,c=(s['panel_'+j] for j in 'abc'); u=a['u']; v=a['v']; good=np.max(abs(u-v),1)<1
 m=np.array([np.linalg.lstsq(aff(u[good][:,j:j+1]),v[good,j],rcond=None)[0] for j in range(4)]); z=(xx-m[:,0])/m[:,1]
 mb=np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]
 u=c['u']; v=c['v1']; w=c['v2']; h=quad(u); p1=np.linalg.lstsq(h,v,rcond=None)[0]
 def v2f(u,r,lin):
  q=np.expm1(-(u-r[4:])**2*np.exp(r[:4])); return np.c_[aff(u),q] if lin else aff(q)
 fits=[]
 for lin in (False,True):
  for k in (.3,1.,3.):
   def fun(r):
    h=v2f(u,r,lin); return (h@np.linalg.lstsq(h,w,rcond=None)[0]-w).ravel()
   opt=least_squares(fun,np.r_[np.full(4,np.log(k)),np.zeros(4)],bounds=([-5]*4+[-1.5]*4,[3]*4+[1.5]*4),max_nfev=220); fits.append((np.std(opt.fun),lin,opt.x))
 er,lin,r=min(fits,key=lambda a:a[0]); h2=v2f(u,r,lin); p2=np.linalg.lstsq(h2,w,rcond=None)[0]
 if debug: print('B',mb,'C1 residual',np.std(h@p1-v),'C1 params',p1,'C2 residual/params',er,lin,r,p2,flush=True)
 return z,aff(z)@mb,quad(z)@p1,v2f(z,r,lin)@p2,mb,p1,p2,r,lin

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; t=s['eval_times']; Y=np.c_[tr['y'],tr['traj']]; n=len(X); Q=-np.expm1(-t[:,None]*[.425,1.832]); A=np.c_[Y[:,0],Y[:,1:]@np.linalg.pinv(Q).T]
 z,b,c,d,mb,p1,p2,r,lin=maps(s,X,True); da={k:h for k,h in zip('zbcd',(z,b,c,d))}; da.update({'c0':quad(b)@p1,'d0':np.expm1(-(b-r[4:])**2*np.exp(r[:4])),'bi':(z-mb[0])@np.linalg.inv(mb[1:]),'zp':np.expm1(-(z-r[4:])**2*np.exp(r[:4]))}); print('target corr',[(k,np.round(np.corrcoef(np.c_[f,A].T)[:4,4:],2)) for k,f in da.items()]);
 folds=[(np.arange(52),np.arange(52,72))]; folds.extend((np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(134).permutation(n),4));
 def ev(K,reg):
  es=[]
  for a,h in folds:
   aa=K[np.ix_(a,a)]; bb=K[np.ix_(h,a)]; mean=aa.mean(0); aa=aa-mean[:,None]-mean+mean.mean(); bb=bb-bb.mean(1)[:,None]-mean+mean.mean(); y=Y[a]; yh=y.mean(0); p=bb@solve(aa+reg*np.eye(len(a)),y-yh,assume_a='pos')+yh; p[:,1:]=p[:,1:]@np.linalg.pinv(Q).T@Q.T; e=abs(p-Y[h]); es.append([e[:,0].mean(),e[:,1:].mean()]);
  return np.array(es)
 rs=[]
 sets=[(k,) for k in da]+list(combinations(da,2))+list(combinations('zbcd',3))+[tuple('zbcd')]
 for keys in sets:
  f=np.column_stack([da[k] for k in keys]); f=(f-f.mean(0))/np.maximum(f.std(0),1e-9); fs={'raw':f,'tanh':np.tanh(f),'sin':np.sin(f)}
  for k,ff in fs.items():
   D=cdist(ff,ff,'sqeuclidean')/ff.shape[1]; G=ff@ff.T/ff.shape[1]
   ker={f'p{n}':(1+G)**n for n in range(1,5)}; ker.update((f'r{n}',np.exp(-n*D)) for n in (.1,.4,1,2));
   for key,K in ker.items():
    for reg in (1e-7,1e-5,.001,.03):
     p=ev(K,reg); rs.append(('+'.join(keys),k,key,reg,p[0],p[1:].mean(0),p.max(0)))
 for j in (0,1):
  for typ in (4,5):
   print('best',typ,j,*sorted(rs,key=lambda x:x[typ][j])[:6], sep='\n')
