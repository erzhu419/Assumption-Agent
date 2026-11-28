import pickle,numpy as np
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X,Y,Z=[S['train'][k] for k in ('x','y','traj')]; t=S['eval_times']
def affine(p,k):
 u=S[p]['u'];return np.linalg.lstsq(np.c_[np.ones(len(u)),u],S[p][k],rcond=None)[0]
def forward(x,a):return x@a[1:]+a[0]
def inverse(x,a):return (x-a[0])@np.linalg.inv(a[1:])
A=np.load('/tmp/align.npy'); B=affine('panel_b','v'); C=affine('panel_c','v1'); T=-np.expm1(-np.array([.40552,1.81221])[:,None]*t); Q=Z@np.linalg.pinv(T); targets=np.c_[Y,Q]; i=np.arange(52); j=np.arange(52,72)
spaces={};
for n,u in [('raw',X),('A',forward(X,A)),('Ai',inverse(X,A))]:
 spaces[n]=u; spaces[n+' B']=forward(u,B); spaces[n+' Bi']=inverse(u,B); spaces[n+' C']=forward(u,C)
funcs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'erf':__import__('scipy').special.erf,'rational':lambda a:a/(1+a*a),'softplus':lambda a:np.logaddexp(a,0)}
def fit(u,q,fun,idx,mode,starts=(1.,),nfev=140):
 def features(p):
  w=np.exp(p[:4])*u+p[4:]; h=fun(w); cols=[np.ones(len(u)),h];
  if 'L' in mode: cols.append(u)
  if 'I' in mode: cols.append(np.column_stack([h[:,a]*h[:,b] for a in range(4) for b in range(a+1,4)]))
  if 'S' in mode: cols.append(h*h)
  return np.column_stack(cols)
 def calc(p):
  D=features(p); w=np.linalg.lstsq(D[idx],q[idx],rcond=1e-9)[0]; return D@w
 scale=np.maximum(.05,np.std(q[idx],axis=0)); best=None
 for start in starts:
  r=least_squares(lambda p:((calc(p)-q)[idx]/scale).ravel(),np.r_[np.repeat(np.log(start),4),np.zeros(4)],bounds=(np.r_[[-2.5]*4,[-3.]*4],np.r_[[1.5]*4,[3.]*4]),max_nfev=nfev)
  if best is None or r.cost<best.cost:best=r
 return best,calc(best.x)
if __name__=='__main__':
 results=[]
 for n,u in spaces.items():
  for f,fn in funcs.items():
   for mode in ('','L','I','LI','LIS'):
    r,p=fit(u,targets,fn,i,mode,(1.,1.8) if f in ('sin','rational') else (1.,),90)
    e=np.column_stack([p[:,0]-Y,(p[:,1:]-Q)@T]); tr=np.mean(abs(e[i]),axis=0); va=np.mean(abs(e[j]),axis=0); results.append((n,f,mode,tr[0],tr[1:].max(),va[0],va[1:].max(),r.x))
 np.set_printoptions(precision=3,suppress=True)
 for col in (5,6,3,4):
  print('SORT COLUMN',col)
  for row in sorted(results,key=lambda a:a[col])[:7]: print(row[:3],np.array(row[3:7]),row[7])
 print('Individual response searches in the plausible B coordinate paths:'); out=[]
 for n in ('raw B','Ai B','A B'):
  for c in range(3):
   for f in ('tanh','sin','rational'):
    for m in ('','L','I','LI'):
     r,p=fit(spaces[n],targets[:,c:c+1],funcs[f],i,m,(.8,1.6),140);e=np.abs(p[:,0]-targets[:,c]);out.append((c,e[i].mean(),e[j].mean(),n,f,m,r.x))
 for c in range(3):
  for k in (1,2):
   print(c,k,sorted([a for a in out if a[0]==c],key=lambda a:a[k])[:3])
