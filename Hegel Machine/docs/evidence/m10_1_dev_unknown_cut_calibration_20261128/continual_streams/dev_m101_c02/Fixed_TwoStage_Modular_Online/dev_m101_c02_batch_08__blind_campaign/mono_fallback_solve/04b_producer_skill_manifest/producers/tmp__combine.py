from links import *
from scipy.linalg import qr

def get(s,x):
 z=inv(x,calfit(s)); D={'z':z,'x':x};
 for p,n,f,lim in [('panel_b','v','rat',(.2,2)),('panel_c','v1','soft',(.3,2)),('panel_c','v2','sqrat',(.03,.7))]:
  u=s[p]['u'];v=s[p][n];fn=families[f]
  def feat(x,k):return bias(np.c_[x,fn(x,k)])
  def err(k):
   a=feat(u,k);return np.mean((a@ls(a,v)-v)**2)
  k=minimize_scalar(err,bounds=lim,method='bounded').x
  coef=ls(feat(u,k),v)
  for key,a in list(D.items()):
   if len(key)<3:D[key+n]=feat(a,k)@coef
  print(n,round(k,5),round(err(k)**.5,6))
 return D

def fitridge(x,y,xt,l):
 m=x.mean(0);d=np.maximum(x.std(0),1e-10);a=(x-m)/d;r=(xt-m)/d
 K=a@a.T
 return y.mean(0)+r@a.T@np.linalg.solve(K+l*np.eye(len(K)),y-y.mean(0))

def report(D,Y):
 rng=np.random.default_rng(124); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(5)]; out=[]
 def score(n,F,l):
  errs=[]
  for sp in splits:
   i,j=sp[:56],sp[56:];p=fitridge(F[i],Y[i],F[j],l);errs.append(np.abs(p-Y[j]).mean(0))
  e=np.array(errs);out.append((n,l,e.mean(0),e.max(0),e[0]))
 sets=[(k,[k]) for k in D]+[('all',list(D)),('root',['z','zv','zv1','zv2']),('rootx',['x','xv','xv1','xv2'])]+[('+'.join(k),k) for k in combinations(['z','zv','zv1','zv2'],2)]
 for n,keys in sets:
  for d in ['lin','univar','poly2','poly3','uni+cross']:
   b=[]
   for k in keys:
    z=D[k]; p,nms=poly(z,3)
    b.append({'lin':z,'univar':np.c_[z,z**3,z**2,z/(1+z*z),z/(1+abs(z)),z*z/(1+.18*z*z)],'poly2':p[:,1:15],'poly3':p[:,1:],'uni+cross':np.c_[z,z**3,z/(1+z*z),z/(1+abs(z)),z*z/(1+.18*z*z),p[:,5:15]]}[d]);
   for l in [.001,.03,.3,2,8]:score(n+' '+d,np.concatenate(b,axis=1),l)
 for j in range(2):
  print('target',j);print(*[(n,l,np.round(a,4),np.round(b,4),np.round(c,4)) for n,l,a,b,c in sorted(out,key=lambda r:r[2][j])[:12]],sep='\n')

if __name__=='__main__':
 tr=s['train']; T=tr['traj'];_,_,v=np.linalg.svd(T,full_matrices=False); D=get(s,tr['x']); print('keys',list(D)); print('statistics y/t',tr['summary'],np.mean(tr['y']),np.std(tr['y']),np.mean(T[:,-1]),np.std(T[:,-1]));report(D,np.column_stack((tr['y'],T@v[0]*v[0,-1])))
 for k in ['z','zv','zv1','zv2']:
  z=D[k]; print(k,'ranges',z.min(0),z.max(0));f,n=poly(z,3)
  from probe import sparse
  for y in [tr['y'],T@v[0]*v[0,-1],T@v[1],T@v[2]]:sparse(f,y,n,5,6)
