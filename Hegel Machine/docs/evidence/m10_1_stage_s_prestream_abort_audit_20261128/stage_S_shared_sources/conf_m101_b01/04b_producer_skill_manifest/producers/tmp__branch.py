exec(open('/tmp/discover.py').read().split('U,D,V=')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
from collections import defaultdict
np.set_printoptions(precision=4,suppress=True,linewidth=145)
fn={'s':np.sin,'c':np.cos,'t':np.tanh,'r':lambda a:a/(1+a*a),'p':lambda a:a*a,'o':lambda a:a**3}
pc=s['panel_c']; u=pc['u']; folds=[np.arange(24)%4==i for i in range(4)]
def base(X,p,seq,lin):
 return np.column_stack([np.ones(len(X))]+([X] if lin else [])+[fn[f](w*X) for w,f in zip(p,seq)])
def fit(X,Y,idx,seq,lin,init):
 def solve(p):
  D=base(X[idx],p,seq,lin); w=np.linalg.lstsq(D,Y[idx],rcond=None)[0]; return D@w-Y[idx],w
 if seq:p=least_squares(lambda p:solve(p)[0].ravel(),init,bounds=(.3,6),max_nfev=160).x
 else:p=[]
 e,w=solve(p);return p,w,np.mean(abs(e))
models={}
for key,seqs in [('v1',['t','r','s','ts','tr','to','rs','so','to','tt','rr']),('v2',['c','cp','cc','cs','p'])]:
 Y=pc[key]; scores=[]
 for lin in [False,True]:
  for seq in seqs:
   init={'t':2.2,'c':1.3,'r':3.4,'s':2.5,'o':1.,'p':1.}; start=np.array([init[f] for f in seq]);
   if len(seq)==2 and seq[0]==seq[1]:start[1]*=.7
   p,w,e=fit(u,Y,np.ones(24,dtype=bool),seq,lin,start); cv=[]
   for mask in folds:
    pp,ww,_=fit(u,Y,~mask,seq,lin,start);cv.append(np.mean(abs(base(u[mask],pp,seq,lin)@ww-Y[mask])))
   scores.append((np.mean(cv),max(cv),e,seq,lin,p,w))
 scores.sort(key=lambda row:row[0]); print(key,[(round(v,6),round(m,6),round(e,6),seq,lin,p) for v,m,e,seq,lin,p,_ in scores[:7]])
 for _,_,_,seq,lin,p,w in scores[:2]:
  print('weights',key,seq,lin,w);models[key+str(len([k for k in models if k.startswith(key)]))]=(seq,lin,p,w)
def tb(p):return -np.expm1(-t[:,None]*p)
p=least_squares(lambda p:(tr-tr@np.linalg.pinv(tb(p).T)@tb(p).T).ravel(),[.365,1.908],bounds=(.05,10)).x; print('weighted temporal',p,np.mean(abs(tr-tr@np.linalg.pinv(tb(p).T)@tb(p).T)))
Y=np.c_[y,tr@np.linalg.pinv(tb(p).T)]; q=coords['AinvB']; idx=np.arange(len(q)); folds=[idx>=56]+[idx%4==j for j in range(4)];
C={key:base(q,p,seq,lin)@w for key,(seq,lin,p,w) in models.items()}
raw={'Q':q,**C,'C':np.c_[C['v10'],C['v20']],'QC':np.c_[q,C['v10'],C['v20']]}
raw.update({'Qv1':np.c_[q,C['v10']],'Qv2':np.c_[q,C['v20']]})
raw['shared']=np.column_stack([q,np.tanh(2.2*q),np.cos(1.3*q),np.sin(2.5*q),q/(1+8*q*q)])
for name,M in raw.items():
 for deg in ([1,2,3] if M.shape[1]<=4 else [1,2]):
  D=np.column_stack([np.prod(M[:,j],axis=1) for k in range(1,deg+1) for j in cwr(range(M.shape[1]),k)]); out=[]
  for mask in folds:
   h=~mask; mx=D[h].mean(0); sx=D[h].std(0); Z=(D-mx)/sx; yc=Y[h].mean(0); U,S,V=np.linalg.svd(Z[h],full_matrices=False); proj=U.T@(Y[h]-yc); vals=[]
   for lam in [1e-8,1e-5,1e-3,.03,.3,3]:
    pred=Z[mask]@(V.T@((S/(S*S+lam))[:,None]*proj))+yc; e=pred-Y[mask]; vals.append(np.r_[np.mean(abs(e),axis=0),np.mean(abs(e[:,1:]@tb(p)[-1]))])
   out.append(vals)
  out=np.array(out); m=out.mean(0); ids=np.argmin(m,axis=0); print(name,deg,'mean',m[ids,np.arange(4)],'worst',out.max(0)[ids,np.arange(4)],'ridge',ids)
