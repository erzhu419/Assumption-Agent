import numpy as n,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];n.set_printoptions(precision=5,suppress=True,linewidth=155)
def ls(a,b):return n.linalg.lstsq(a,b,rcond=1e-12)[0]
def aff(x):return n.c_[n.ones(len(x)),x]
def fit(a,b):return n.sqrt(n.mean((a@ls(a,b)-b)**2,axis=0))
A=s['panel_a']; ba=ls(aff(A['u']),A['v']); keep=n.argsort(n.linalg.norm(aff(A['u'])@ba-A['v'],axis=1))[:45];ba=ls(aff(A['u'][keep]),A['v'][keep]);B=s['panel_b'];bb=ls(aff(B['u']),B['v']);C=s['panel_c'];u=C['u'];
func={'sin':n.sin,'tanh':n.tanh,'cos':n.cos,'exp2':lambda x:n.exp(-x*x),'rat2':lambda x:1/(1+x*x),'rat1':lambda x:x/(1+x*x),'asin':n.arctan,'soft':lambda x:n.sqrt(1+x*x)}
def ev(x,N,P):return n.column_stack([aff(x),*[func[k](p*x) for k,p in zip(N,P)]])
best={}
for key in ('v1','v2'):
 v=C[key];print(key)
 print('diag poly degrees',[(d,fit(n.column_stack([aff(u),*[u**j for j in range(2,d+1)]]),v)) for d in (2,3,4,5)] )
 ranked=[]
 for size in (1,2,3):
  for names in it.combinations_with_replacement(func,size):
   if size==3 and any(q not in ('sin','tanh','cos','rat2','rat1') for q in names):continue
   if len(set(names))<size:continue
   p=least_squares(lambda p:(ev(u,names,p)@ls(ev(u,names,p),v)-v).ravel(),[1.2]*size,bounds=(.15,5),max_nfev=80).x
   X=ev(u,names,p); err=fit(X,v); ranked.append((n.linalg.norm(err),names,p,err,ls(X,v)))
  for score,names,p,e,beta in sorted(ranked,key=lambda r:r[0])[:4]:print(size,names,p,e)
 best[key]=sorted(ranked,key=lambda r:r[0])[:5];print('best coeff',best[key][0][-1].T)
 # rank candidate scalar terms in each column, conditional on affine plus per-coordinate odd/even polynomial
 for j in range(4):
  base=aff(u); terms=[]
  for i in range(4):
   for h in range(i,4):
    z,w=u[:,i],u[:,h];terms.extend([(f'p{i}{h}',z*w),(f'p{i}{i}{h}',z*z*w),(f'p{i}{h}{h}',z*w*w)])
  for idx in it.combinations(range(4),3):terms.append((str(idx),u[:,idx].prod(1)))
  chosen=[]
  for _ in range(5):
   err,k,h=min(((float(fit(n.c_[base,h],v[:,j])),k,h) for k,h in terms if k not in chosen),key=lambda q:q[0]);chosen.append(k);base=n.c_[base,h];
  print('greedy',j,chosen,err)
x=s['train']['x']; t=s['eval_times']; obs=n.c_[s['train']['y'],s['train']['traj']];
G={'raw':x,'Ad':(x-ba[0])/n.diag(ba[1:]),'Af':aff(x)@ba}
for k,g in list(G.items()): G['B'+k]=aff(g)@bb;G['I'+k]=(g-bb[0])@n.linalg.inv(bb[1:]);
ranked=[]
for name,g in G.items():
 P=[aff(g)]; names=['lin'];
 for k in ('v1','v2'):
  for _,fn,p,_,b in best[k][:3]: P.append(ev(g,fn,p)@b);names.append(k+str(fn))
 for cols in it.combinations_with_replacement(range(len(P)),2):
  z=n.column_stack([P[i] for i in set(cols)]+[n.ones(len(g))]);pred=z[56:]@ls(z[:56],obs[:56]);e=n.mean(abs(pred-obs[56:]),0); ranked.append((e[0]+e[1:].mean(),name,[names[i] for i in cols],e[[0,1,4,8]]))
for a in sorted(ranked,key=lambda r:r[0])[:20]:print(a)
