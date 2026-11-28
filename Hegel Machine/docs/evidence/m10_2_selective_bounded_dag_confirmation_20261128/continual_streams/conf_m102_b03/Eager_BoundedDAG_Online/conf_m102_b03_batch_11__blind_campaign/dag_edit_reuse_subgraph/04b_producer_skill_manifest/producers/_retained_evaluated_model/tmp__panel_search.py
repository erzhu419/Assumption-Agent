exec(open('/tmp/explore_cal.py').read().split('np.set_printoptions')[0])
M=ls(ones(s['panel_b']['u']),s['panel_b']['v']); B=M[1:]; h=M[0]; P=np.linalg.inv(B)
print('singular',np.linalg.svd(B)[1]); print('matrix',M)
from scipy.optimize import least_squares
pan=s['panel_c']; u=pan['u']; funcs=[]
for k in (1,2):
 v=pan['v'+str(k)]
 def ft(z,p,k=k):
  w=p[0]*z+p[1:]; return ones(np.c_[z,w/(1+abs(w)) if k==1 else np.cos(w)])
 def fr(p):
  F=ft(u,p);return (F@ls(F,v)-v).ravel()
 q=least_squares(fr,[.84 if k==1 else 1.27,0,0,0,0],max_nfev=150).x; co=ls(ft(u,q),v); funcs.append(lambda z,ft=ft,q=q,co=co: ft(z,q)@co);print('branch',k,q,co)
ops={'I':lambda v:v,'f':lambda v:v@B+h,'r':lambda v:(v-h)@P,'ft':lambda v:v@B.T+h,'rt':lambda v:(v-h)@P.T,'fl':lambda v:v@B,'rl':lambda v:v@P,'flt':lambda v:v@B.T,'rlt':lambda v:v@P.T,'x':lambda v:v*b+a}
C={k:funcs[0](f(z)) for k,f in ops.items()};D={k:funcs[1](f(z)) for k,f in ops.items()}
def eval(F,V):
 r=F[52:]@ls(F[:52],V[:52])-V[52:];return np.mean(abs(r))
ys=[];ts=[]
for i,c in C.items():
 for j,d in D.items():
  dp=(c*d).sum(1); cr=np.einsum('ni,nj->nij',c,d).reshape(len(z),-1); l=np.c_[c,d]
  for name,f in {'base':l,'dot':np.c_[l,dp], 'dot2':np.c_[l,dp,dp**2], 'dot_r':np.c_[l,dp/(1+2.15*dp**2)], 'dot_tanh':np.c_[l,np.tanh(2.15*dp)], 'dot_sin':np.c_[l,np.sin(2.15*dp)], 'cross':np.c_[l,cr], 'quad':np.c_[l,cr,*[w[:,a]*w[:,b] for w in (c,d) for a in range(4) for b in range(a,4)]]}.items():
   F=ones(f);ys.append((eval(F,y),i,j,name)); ts.append((eval(F,t),i,j,name))
print('scalar',sorted(ys)[:25]);print('trajectory',sorted(ts)[:25]);
