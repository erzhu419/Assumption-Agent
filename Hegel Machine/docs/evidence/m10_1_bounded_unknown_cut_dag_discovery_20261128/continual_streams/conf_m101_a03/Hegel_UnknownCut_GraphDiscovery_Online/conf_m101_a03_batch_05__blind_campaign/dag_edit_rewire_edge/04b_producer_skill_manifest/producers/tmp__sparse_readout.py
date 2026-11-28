exec(open('/tmp/probe_readout.py').read().split("for name,q in")[0].split("print('warp'")[0])
from itertools import combinations_with_replacement as cwr
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from scipy.linalg import qr
q=np.c_[x,z]; names=['x'+str(i) for i in range(4)]+['z'+str(i) for i in range(4)]
D={}
def add(name,v): D[name]=v
for k in [2,3]:
 for ids in cwr(range(8),k): add('*'.join(names[i] for i in ids),np.prod(q[:,ids],1))
for i in range(8):
 for j in range(i,8):
  for a,b in [(1,1),(1,-1),(1,0)][:2 if i!=j else 3]:
   if i==j and b: continue
   w=a*q[:,i]+b*q[:,j]; nm=f'{names[i]}{b:+d}{names[j]}'
   for f in [1,2,3,4]:
    add(f'sin{f}({nm})',np.sin(f*w));add(f'cos{f}({nm})',np.cos(f*w));add(f'tanh{f}({nm})',np.tanh(f*w))
  w=q[:,i]*q[:,j]
  for f in [1,2,3]:
   add(f'sin{f}({names[i]}*{names[j]})',np.sin(f*w)); add(f'tanh{f}({names[i]}*{names[j]})',np.tanh(f*w))
  for f in [.5,1,2]: add(f'{names[i]}/(1+{f}{names[j]}^2)',q[:,i]/(1+f*q[:,j]**2)); add(f'{names[j]}/(1+{f}{names[i]}^2)',q[:,j]/(1+f*q[:,i]**2))
F=np.column_stack(list(D.values())); N=np.array(list(D)); idx=np.arange(n)
folds=[np.arange(52,72)]+[v for t,v in KFold(5,shuffle=True,random_state=18).split(x)]
for base in [np.c_[np.ones(n),z],np.c_[np.ones(n),x,z]]:
 E=[]
 for v in folds:
  t=np.setdiff1d(idx,v); B=base[t]; resid=y[t]-B@np.linalg.lstsq(B,y[t],rcond=None)[0]; R=F[t]-B@np.linalg.lstsq(B,F[t],rcond=None)[0]; norm=np.linalg.norm(R,axis=0); valid=norm>1e-5; norm[~valid]=1; R/=norm; R[:,~valid]=0
  order=np.argsort(-np.abs(resid@R)); selected=[]; es=[]
  for k in range(8):
   j=np.argmax(np.abs(resid@R)); selected.append(j); R[:,j]=0; A=np.c_[base,F[:,selected]]; w=np.linalg.lstsq(A[t],y[t],rcond=None)[0]; resid=y[t]-A[t]@w; es.append(np.abs(A[v]@w-y[v]).mean())
  E.append(es); print('base',B.shape[1], 'first',N[order[:7]], 'greedy',N[selected[:5]],np.round(es,4))
 print('worst',np.max(E,0).round(4))
