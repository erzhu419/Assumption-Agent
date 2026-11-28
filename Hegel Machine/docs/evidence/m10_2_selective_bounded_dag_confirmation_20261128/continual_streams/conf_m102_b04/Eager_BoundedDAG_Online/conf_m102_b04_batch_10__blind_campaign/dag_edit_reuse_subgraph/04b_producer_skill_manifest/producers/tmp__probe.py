import pickle,numpy as np
from itertools import combinations_with_replacement as C
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];x=tr['x'];Y=np.column_stack([tr['y'],tr['traj']]);u,v=s['panel_a'].values(); b=s['panel_b']; c=s['panel_c'];
def P(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in C(range(4),k)])
def ls(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
def robust(f,v):
 keep=np.linalg.norm(v-f@ls(f,v),axis=1)<3
 for _ in range(10):
  e=np.linalg.norm(v-f@ls(f[keep],v[keep]),axis=1);keep=e<max(.09,min(1.,np.median(e)*3))
 return ls(f[keep],v[keep]),keep
print('panel A residuals by component');h,k=robust(P(u,1),v);print(k.sum(),h.round(3));print(np.c_[u[k],(v-P(u,1)@h)[k]].round(3)[:25]);
for name,w,z in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2'])]:
 print(name);h2=ls(P(w,2),z); print(h2.round(3));print('fit',np.sqrt(np.mean((P(w,2)@h2-z)**2,axis=0)))
xa=(x-h[0])@np.linalg.inv(h[1:]); xd=(x-h[0])/h[1:].diagonal(); folds=[np.arange(j,j+20)%72 for j in (0,18,36,52)];
def evaluate(K,a):
 out=[]
 for j in folds:
  i=np.setdiff1d(np.arange(72),j); z=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+a*np.eye(len(i)),Y[i],assume_a='pos');e=abs(z-Y[j]);out.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(out)
results=[]
for nam,w in [('x',x),('xa',xa),('xd',xd)]:
 sets=[(nam,w)];
 for n,p,k in [('B',b,'v'),('C1',c,'v1'),('C2',c,'v2')]:
  for d in (1,2):
   z=P(w,d)@ls(P(p['u'],d),p[k]);sets.extend([(nam+n+str(d),z),(nam+'+'+n+str(d),np.c_[w,z])]);
 for n,z in sets:
  for kernel in ('rbf','matern','poly2','poly3','poly4','poly6'):
   for l in (.45,.8,1.4,2.5,4.):
    D=np.sum((z[:,None,:]-z[None,:,:])**2,axis=2)/l**2
    if kernel=='rbf': K=np.exp(-D/2)+1
    elif kernel=='matern': D=np.sqrt(5*D);K=(1+D+D**2/3)*np.exp(-D)+1
    else:K=(1+z@z.T/l**2)**int(kernel[-1]);K/=np.mean(np.diag(K))
    for a in (1e-5,.0003,.006,.05):
     E=evaluate(K,a);results.append((E,n,kernel,l,a));
for metric in [lambda e:e.mean(),lambda e:e[:,0].mean(),lambda e:e[:,1].mean()]:
 for E,*conf in sorted(results,key=lambda z:metric(z[0]))[:7]:print(conf,E.T.round(4))
