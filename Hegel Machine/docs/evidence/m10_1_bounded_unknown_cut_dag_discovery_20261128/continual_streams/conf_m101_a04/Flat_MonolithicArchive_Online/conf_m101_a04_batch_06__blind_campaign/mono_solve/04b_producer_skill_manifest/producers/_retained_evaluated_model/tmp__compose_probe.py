import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement,combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
def cal(panel,raw):
 u,v=panel['u'],panel['v'];k=np.linalg.norm(v,axis=1)<5;u,v=u[k],v[k];basis=lambda x:np.stack([np.ones_like(x),x,x**3,(x-1)*np.roll(x,-1,axis=1)],axis=2);a=basis(u);w=np.stack([fit(a[:,i],v[:,i]) for i in range(4)]);pred=lambda x:(basis(x)*w).sum(2);x=np.stack([least_squares(lambda z:(pred(z[None])-r).ravel(),r,gtol=1e-12).x for r in raw]);print('cal inverse max',np.max(abs(pred(x)-raw)));return x
def bil(x):
 a,b,c,d=x.T;return bias(np.c_[a*b,a*d,b*c,c*d])
def cos(x,w):return bias(np.c_[x,np.cos(w*x)])
def operators(s,A):
 pb,p=s['panel_b'],s['panel_c'];u=p['u'];wb=fit(bias(pb['u']),pb['v']);wc=fit(bil(u),p['v1']); f=lambda w:np.mean((cos(u,w)@fit(cos(u,w),p['v2'])-p['v2'])**2);w=minimize_scalar(f,bounds=(1.9,2.9),method='bounded').x;wd=fit(cos(u,w),p['v2']); return (lambda x:bias(x)@wb,lambda x:bil(x)@wc,lambda x:cos(x,w)@wd)
tr=s['train']; raw=tr['x'];x=cal(s['panel_a'],raw);B,C,D=operators(s,x); y=tr['y'];z=tr['traj'];t=s['eval_times'];n=52
A={'r':raw,'x':x,'b':B(x),'c':C(x),'d':D(x),'j':C(B(x)),'k':D(B(x)),'l':B(C(x)),'m':B(D(x)),'p':C(D(x)),'q':D(C(x))}
def metrics(F,T):
 out=F[n:]@fit(F[:n],T[:n]);return np.mean(abs(out-T[n:])),np.sqrt(np.mean((F[:n]@fit(F[:n],T[:n])-T[:n])**2))
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
if __name__=='__main__':
 scores=[]
 for i in A:
  for d in [1,2,3]:
   F=poly(A[i],d);scores.append((i+str(d),*metrics(F,y),*metrics(F,z)))
 for ni in [2,3]:
  for keys in combinations(A,ni):
   for mode in [0,1,2,3]:
    ar=[A[k] for k in keys]; fs=ar.copy()
    if mode in (1,3):fs.extend(a*b for a,b in combinations(ar,2));fs.extend(np.sum(a*b,axis=1) for a,b in combinations(ar,2))
    if mode in (2,3):fs.extend(poly(a,2)[:,5:] for a in ar)
    F=bias(np.column_stack(fs));scores.append((''.join(keys)+':'+str(mode),*metrics(F,y),*metrics(F,z)))
 for k in [1,3]:print('top',k,sorted(scores,key=lambda a:a[k])[:14]);print('by train',k,sorted(scores,key=lambda a:a[k+1])[:8])
 z=z[:52];fs={'e':lambda k:-np.expm1(-k*t),'r':lambda k:t/(1+k*t),'d':lambda k:t*np.exp(-k*t),'s':lambda k:np.sin(k*t),'q':lambda k:t/(1+k*t*t),'t':lambda k:np.tanh(k*t)}
 for L in [2,3]:
  ls=[]
  for names in combinations_with_replacement(fs,L):
   def design(p):return np.stack([fs[k](v) for k,v in zip(names,p)],axis=1)
   def res(p):
    H=design(p);return (H@fit(H,z.T)-z.T).ravel()
   p=least_squares(res,np.linspace(.4,1.7,L),bounds=(.01,4),max_nfev=90).x;ls.append((np.std(res(p)),''.join(names),p))
  print('time',L,sorted(ls,key=lambda a:a[0])[:12])
