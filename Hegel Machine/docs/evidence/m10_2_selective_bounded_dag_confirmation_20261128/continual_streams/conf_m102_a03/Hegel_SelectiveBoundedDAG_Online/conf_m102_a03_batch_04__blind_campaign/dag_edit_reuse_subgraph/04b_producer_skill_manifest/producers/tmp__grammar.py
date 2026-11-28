import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from features import *
from scipy.optimize import least_squares
from itertools import combinations
FNS=[('cube',lambda q:q**3),('tanh',np.tanh),('sin',np.sin),('rational',lambda q:q/(1+q*q)),('quad',lambda q:q*q)]
def gram(q):
 n,d=q.shape; i,j=np.triu_indices(d,1)
 return dict(aff=poly(q),sq=q*q,inter=q[:,i]*q[:,j],cube=q**3,cyc=q*np.roll(q,-1,axis=1),sin=np.sin(q),tanh=np.tanh(q),oddmix=q*q*np.roll(q,-1,axis=1),trip=np.column_stack([np.prod(q[:,p],axis=1) for p in combinations(range(d),3)]))
mods=[('aff','cube'),('aff','sq','cube'),('aff','cyc','cube'),('aff','sq','cube','inter'),('aff','trip'),('aff','trip','cube'),('aff','cyc','oddmix'),('aff','sq','inter'),('aff','cyc'),('aff','sin','cyc')]
def panel_fit(U,V,Z):
 G,H=gram(U),gram(Z); res=[]
 for names in mods:
  X=np.column_stack([G[k] for k in names]);A=np.column_stack([H[k] for k in names]);P=np.linalg.pinv(X);C=P@V;e=(X@C-V)/np.maximum(1-np.sum(X*P.T,axis=1),1.e-8)[:,None];res.append((np.mean(abs(e)),names,A@C))
 for f_name,fn in FNS:
  for add in [('aff',),('aff','sq'),('aff','cyc'),('aff','cube')]:
   X=np.column_stack([G[k] for k in add]);A=np.column_stack([H[k] for k in add]); pred=[];ee=[]
   for v in V.T:
    def fit(w):
     D=np.c_[X,fn(G['aff']@w)];C=ls(D,v);return D,C
    start=ls(G['aff'],v);start/=max(np.linalg.norm(start[1:]),.1)
    opts=[]
    for scale in [.65,1.5,2.5]:
     o=least_squares(lambda w:(lambda dc:dc[0]@dc[1]-v)(fit(w)),scale*start,max_nfev=180);opts.append(o)
    w=min(opts,key=lambda o:o.cost).x;D,C=fit(w);pred.append(np.c_[A,fn(H['aff']@w)]@C);hh=np.sum(D*np.linalg.pinv(D).T,axis=1);ee.append(np.mean(abs((D@C-v)/np.maximum(1-hh-.15,.06))))
   res.append((np.mean(ee), (f_name,)+add,np.column_stack(pred)))
 return sorted(res,key=lambda x:x[0])
if __name__=='__main__':
 fs=[]
 for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  U=s[p]['u'];V=s[p][k];rr=panel_fit(U,V,z);print(p,k,[(r[1],round(r[0],5)) for r in rr[:10]],flush=True)
  fs.append(rr)
  for r in rr[:4]:chk(str(r[1]),poly(r[2],2))
 for i in range(4):
  for j in range(4):
   for k in range(2):
    g,b,c=fs[0][i][2],fs[1][j][2],fs[2][k][2];f=np.c_[np.ones(len(z)),g,b,c];print(i,j,k,score(f).max(axis=0),score(np.c_[f,g*b,g*c,b*c]).max(axis=0))
