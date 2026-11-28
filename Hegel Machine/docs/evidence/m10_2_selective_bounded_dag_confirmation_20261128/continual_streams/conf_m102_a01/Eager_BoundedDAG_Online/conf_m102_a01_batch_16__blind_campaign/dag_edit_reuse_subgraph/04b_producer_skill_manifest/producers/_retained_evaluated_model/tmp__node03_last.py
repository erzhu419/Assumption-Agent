import pickle,numpy as np
from scipy.optimize import minimize
from itertools import combinations
L=pickle.load(open('/tmp/node03_loo.pkl','rb'))
N=pickle.load(open('/tmp/node03_nested.pkl','rb'))

def encode(rows):
 out=[]
 for p,b,vt,y,t in rows:
  s=b[:,0]; d=p[:,0]-s; z=(.4*p[:,1:]+.6*b[:,1:])@vt; E=(p[:,1:]-b[:,1:])@vt
  a=z[:,-1]; k=z[:,0]-.2*a; f=E[:,-1]; g=E[:,0]-.2*f
  out.append(np.column_stack([s,d,a,k,f,g,y]))
 return np.vstack(out)

def feat(arr):
 s,d,a,k,f,g,y=arr.T
 return dict(i=np.ones_like(s),d=d,s=s,a=a,k=k,f=f,g=g, s2=s*s,s3=s**3, s4=s**4, asym=abs(s), cap=np.clip(s,-.7,.7), over=s-np.clip(s,-.7,.7),sa=s*a,sk=s*k,ak=a*k,aa=a*a,kk=k*k,df=d*f,dg=d*g,ds=d*s,da=d*a,dk=d*k, ds2=d*s*s,ad=abs(d),af=abs(f),ag=abs(g),f2=f*f,g2=g*g,sd=s*np.abs(d),asq=s*np.abs(s),uf=f/(.025+abs(f)),ug=g/(.006+abs(g)),ud=d/(.04+abs(d)),sp=np.maximum(s,0),sm=np.minimum(s,0),dp=np.maximum(d,0),dm=np.minimum(d,0), sf=s*f,sg=s*g, ga=g*a, fa=f*a)

data=[(encode(tr),encode([v])) for tr,v in L]; verify=[encode([v]) for _,v in N]
features=['i d','i d s','i d s s2 s3','i d s a k f g','i d s2 a k f g','i d a k f g sd asq','i d sp sm dp dm','i d s2 asq dp dm','i d sd asq f g df dg','i d s2 s3 a k a k sa sk ak aa kk','i d s2 s3 a k f g sd asq df dg ds da dk ds2 ad af ag f2 g2 uf ug ud sf sg ga fa']
for group in list(features):
 for key in ('s2','asym','f','g','ds2','df','ak','over'):
  x=' '.join(dict.fromkeys((group+' '+key).split()))
  if x not in features: features.append(x)
models=[]
for group in features:
 keys=group.split()
 cols=lambda A: np.column_stack([feat(A)[k] for k in keys])
 for lam in (1,5,15,45):
  for frac in (0,.35,.65):
   scores=[]
   for tr,va in data:
    X=cols(tr); scale=np.maximum(np.sqrt((X**2).mean(0)),1e-9); X=X/scale; e=tr[:,-1]-tr[:,0]-frac*tr[:,1]
    A=X.T@X+lam*np.eye(len(keys)); w=np.linalg.solve(A,X.T@e)
    for _ in range(5):
     r=X@w-e; wt=1/np.sqrt(1+(r/.035)**2); w=np.linalg.solve(X.T@(X*wt[:,None])+lam*np.eye(len(keys)),X.T@(wt*e))
    w=w/scale
    scores.append(np.mean(abs(va[:,0]+frac*va[:,1]+cols(va)@w-va[:,-1])))
   models.append((scores,group,lam,frac,w))
for rank in (lambda a:np.mean(a[0][:4])+.35*max(a[0][:4]),lambda a:a[0][-1]):
 print('RANK')
 for scores,group,lam,frac,w in sorted(models,key=rank)[:6]: print(np.round(scores,5),repr(group),lam,frac,np.round(w,6))
print('passing',sum(m[0][-1]<=.038 for m in models))
pickle.dump(models,open('/tmp/node03_last_models.pkl','wb'))
