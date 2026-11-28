import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations
from sklearn.linear_model import Lasso
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; t=s['eval_times']; T=s['train']['traj'];y=s['train']['y']; N=52
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
D=lambda *a:np.column_stack([np.ones(len(a[0])),*a])
U=-np.expm1(-t[:,None]*np.array([.41837,1.81795])); H=ls(U,T.T).T; Y=np.c_[y,H]; scale=Y[:N].std(0); Y=Y/scale
p=s['panel_c'];u=p['u'];
funcs={'oddexp':lambda x,k:-np.sign(x)*np.expm1(-k*abs(x)),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'even':lambda x,k:x*x/np.sqrt(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'gauss':lambda x,k:np.exp(-k*x*x)}
ops=[]
for name in ['v1','v2']:
 v=p[name]; cand=[]
 for n,f in funcs.items():
  for lin in [False,True]:
   df=lambda x,k:D(x,f(x,k)) if lin else D(f(x,k)); loss=lambda k:np.linalg.norm(df(u,k)@ls(df(u,k),v)-v); o=minimize_scalar(loss,bounds=(.02,4),method='bounded'); cand.append((o.fun,n,lin,o.x))
 cand.sort(); print(name,cand[:6]);_,n,lin,k=cand[0];f=funcs[n]; df=lambda x,k,f=f,lin=lin:D(x,f(x,k)) if lin else D(f(x,k));w=ls(df(u,k),v);ops.append(lambda x,df=df,k=k,w=w:df(x,k)@w)
b=s['panel_b'];B=ls(D(b['u']),b['v']); blocks={'x':x,'b':D(x)@B,'bi':(x-B[0])@np.linalg.inv(B[1:])};
pa=s['panel_a']; a=pa['u']; v=pa['v']; A=lambda a:D(a,a*a,a**3,np.array([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]).T);F=A(a); W=ls(F,v)
for _ in range(70):
 q=np.maximum(np.linalg.norm(v-F@W,axis=1),.00025)**-.5;W=ls(F*q[:,None],v*q[:,None]);
blocks['a']=A(x)@W
from scipy.optimize import root
from itertools import product
np.set_printoptions(precision=3,suppress=True)
print('Y quantiles',np.quantile(Y,[0,.05,.25,.5,.75,.95,1],axis=0),'Y correlation',np.corrcoef(Y.T)); print('calibration constant/linear',W[:5]);print('x covariance',np.cov(x.T))
ai=np.array([root(lambda u:(A(u[None])@W-r).ravel(),r).x for r in x]); print('inverse residual',np.max(abs(A(ai)@W-x)), 'range',np.quantile(ai,[0,.5,1],axis=0));
cfg={'x':x,'b':blocks['b'],'bi':blocks['bi'],'a':blocks['a'],'ai':ai}
cfg["bai"]=D(ai)@B;cfg["biai"]=(ai-B[0])@np.linalg.inv(B[1:]);
for name,z in list(cfg.items()):
 c,d=(op(z) for op in ops);cfg['c'+name]=c;cfg['d'+name]=d;cfg['m'+name]=np.c_[z,c,d];cfg['cd'+name]=np.c_[c,d]
from itertools import combinations,combinations_with_replacement
from scipy.linalg import eigh
z,c,d=(cfg[n] for n in ['bai','cbai','dbai']); V=np.c_[z,c,d]; names=[p+str(i) for p in 'zcd' for i in range(4)]; feats={};
def add(n,f):feats[n]=f
for n,v in zip(names,V.T):
 add(n,v);add(n+'^2',v*v);add(n+'^3',v**3);add('abs('+n+')',abs(v))
for i,j in combinations(range(12),2):
 a,b=V[:,i],V[:,j];ni,nj=names[i],names[j];add(ni+'*'+nj,a*b);add(ni+'^2*'+nj,a*a*b);add(ni+'*'+nj+'^2',a*b*b)
for i,j,k in combinations(range(12),3):add('*'.join(names[q] for q in (i,j,k)),V[:,i]*V[:,j]*V[:,k])
for i,j in combinations_with_replacement(range(12),2):
 a,b=V[:,i],V[:,j];ni,nj=names[i],names[j]
 for g in ([0.] if i==j else [-1.5,-1.,-.5,.5,1.,1.5]):
  v=a+g*b;n=ni if g==0 else f'({ni}{g:+g}*{nj})'
  for k in [.5,1.,1.5,2.]:
   for key,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('odd',lambda q:-np.sign(q)*np.expm1(-abs(q))),('inv',lambda q:1/(1+q*q)),('rat',lambda q:q/(1+q*q))]:add(f'{key}({k:g}*{n})',f(k*v))
 for k in [.3,.6,1.,1.6]:
  add(f'{ni}/(1+{k}*{nj}^2)',a/(1+k*b*b));add(f'{nj}/(1+{k}*{ni}^2)',b/(1+k*a*a))
  if i!=j:
   for l in range(12):
    if l in (i,j) or i//4==j//4: add(f'{ni}*{nj}/(1+{k}*{names[l]}^2)',a*b/(1+k*V[:,l]**2))
F=np.column_stack(list(feats.values())); keys=np.array(list(feats));
def beam(Y,train,test,linear):
 G=F-F[train].mean(0);G/=np.maximum(np.linalg.norm(G[train],axis=0),1e-12);B=np.c_[np.ones(len(V)),V] if linear else np.ones((len(V),1)); q=np.linalg.svd(B[train],full_matrices=False)[0];a=G[train]; A=a-q@(q.T@a); AA=np.einsum('ij,ij->j',A,A); h=Y[train]-q@(q.T@Y[train]); beam=[()]
 for depth in range(1,9):
  nxt={}
  for idx in beam:
   if idx:
    q2=np.linalg.qr(A[:,idx],mode='reduced')[0];r=h-q2@(q2.T@h); proj=q2.T@A; norms=AA-np.einsum('ij,ij->j',proj,proj)
   else:r=h;norms=AA
   gains=np.divide((r@A)**2,norms,out=np.zeros_like(norms),where=norms>1e-6)
   for j in np.argpartition(gains,-24)[-24:]:
    if j not in idx:nxt[tuple(sorted((*idx,j)))]=r@r-gains[j]
  beam=sorted(nxt,key=nxt.get)[:24];idx=beam[0];M=np.c_[B,G[:,idx]];e=Y-M@ls(M[train],Y[train]);print(depth,np.round([abs(e[train]).mean(),abs(e[test]).mean()],5),list(keys[list(idx)]),flush=True)
for j,yy in enumerate(np.c_[y,H].T):
 for linear in [False,True]:
  print('TARGET',j,'linear',linear,flush=True);beam(yy,np.arange(52),np.arange(52,72),linear)
