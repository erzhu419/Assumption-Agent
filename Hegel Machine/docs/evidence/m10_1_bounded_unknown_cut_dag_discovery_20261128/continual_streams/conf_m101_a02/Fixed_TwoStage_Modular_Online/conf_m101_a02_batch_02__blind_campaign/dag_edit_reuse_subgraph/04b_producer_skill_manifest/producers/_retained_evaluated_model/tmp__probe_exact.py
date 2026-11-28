import numpy as np, pickle, runpy
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; frozen=runpy.run_path(P+'_frozen_nodes/node_00.py')['run_node']; z=frozen({'__input__':s})['z_tr']; c=runpy.run_path(P+'node_01.py'); L=c['_affine'](z); Q=c['_quadratic'](z); C=Q[:,[5+2,5+6]]; Zs=np.column_stack((z,C)); ls=np.linalg.lstsq(L,y,rcond=None)[0][1:]
rng=np.random.default_rng(133); ix=np.arange(len(y)); groups=[ix[-20:]]+[rng.permutation(ix)[:20] for _ in range(16)]+[np.argsort(abs(v))[-12:] for v in z.T]
funcs={'tanh':np.tanh,'atan':np.arctan,'rat':lambda u:u/(1+u*u),'sqrt':lambda u:u/np.sqrt(1+u*u),'sin':np.sin,'asinh':np.arcsinh}
def experiment(base, X, kind, tr, te):
    proj=funcs[kind]; U=base[tr]; Xt=X[tr]; Y=y[tr]; O=np.linalg.qr(U,mode='reduced')[0]; Y0=Y-O@(O.T@Y)
    def res(p):
        v=proj(Xt@p[:-1]+p[-1]); v=v-O@(O.T@v); return v*(v@Y0/max(v@v,1e-25))-Y0
    starts=[np.r_[a*ls,np.zeros(X.shape[1]-4),0.] for a in (.5,1.,1.6)]
    r=min((least_squares(res,p,max_nfev=90) for p in starts),key=lambda r:r.cost)
    H=np.column_stack((base,proj(X@r.x[:-1]+r.x[-1]))); b=np.linalg.lstsq(H[tr],Y,rcond=None)[0]
    return abs(H[te]@b-y[te]).mean(),r.x,b
for bn,B in [('full',Q),('cross',np.column_stack((L,C))),('pairs',Q[:,[0,1,2,3,4,6,7,8,10,11,13]])]:
 for xn,X in [('z',z), ('zC',Zs)]:
  for k in funcs:
   train,p,b=experiment(B,X,k,ix,ix); e=[experiment(B,X,k,np.setdiff1d(ix,g),g)[0] for g in groups]
   print(bn,xn,k,np.round([train,e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],5),'p',p.round(3),'b',b.round(3),flush=True)
print('candidate split scores',np.round([np.mean(abs(c['_label'](z[tr],z[g],y[tr])-y[g])) for g in groups for tr in [np.setdiff1d(ix,g)]],5))
