exec(open('/tmp/graph_probe.py').read().split("for key in ('v1','v2'):")[0])
from scipy.special import expit
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
Y=s['train']['y']; n=len(Y); rg=np.random.default_rng(18); F=[np.arange(52,72)]+list(np.array_split(rg.permutation(n),4)); k=np.arange(n)
p=s['panel_a']; AA=np.c_[np.ones(len(p['u'])),p['u']]; keep=np.ones(len(AA),bool)
for _ in range(6):
 ca=np.linalg.lstsq(AA[keep],p['v'][keep],rcond=None)[0]; r=np.linalg.norm(AA@ca-p['v'],axis=1); keep=r<max(.001,4*np.median(r))
z=(raw-ca[0])@np.linalg.inv(ca[1:]); coords={'raw':raw,'cal':x,'invA':z,'invAB':(z-b[0])@np.linalg.inv(b[1:]),'invBA':((raw-b[0])@np.linalg.inv(b[1:])-ca[0])@np.linalg.inv(ca[1:])}
def lin(D,t):return np.linalg.lstsq(D,t,rcond=1e-9)[0]
report=[]
for nm,w in coords.items():
 w=w[:n]; A=np.c_[np.ones(n),w]; Q=np.array([w[:,i]*w[:,j] for i,j in cr(range(4),2)]).T; B=np.c_[A,Q]; R=np.sum(w*w,axis=1)
 c=lin(B,Y); H=np.zeros((4,4))
 for (i,j),v in zip(cr(range(4),2),c[5:]): H[i,j]=H[j,i]=v*(1 if i==j else .5)
 e,V=np.linalg.eigh(H); print(nm,'linear',c[1:5].round(3),'eig',e.round(3),'linear rotated',(V.T@c[1:5]).round(3),'axes',V.round(2).T.tolist())
 for kind in ('sin','cos','tanh','softplus','rational','gauss','power','radial','radial+ridge','quad+2ridge'):
  for baseline in ('full','iso'):
   if kind in ('radial','radial+ridge','quad+2ridge') and baseline=='full': continue
   D0=B if baseline=='full' or kind=='quad+2ridge' else np.c_[A,R]
   errs=[]; params=[]; rms=[]
   for test in F:
    train=np.setdiff1d(k,test); t=Y[train]; qq=lin(B[train],t); hh=np.zeros((4,4))
    for (i,j),v in zip(cr(range(4),2),qq[5:]):hh[i,j]=hh[j,i]=v*(1 if i==j else .5)
    ee,vv=np.linalg.eigh(hh); d=qq[1:5]/np.linalg.norm(qq[1:5]);
    def feature(p):
     f=w@p[:4]+p[4]
     if kind=='sin': phi=np.sin(f)
     elif kind=='cos': phi=np.cos(f)
     elif kind=='tanh': phi=np.tanh(f)
     elif kind=='softplus': phi=np.logaddexp(0,f)
     elif kind=='rational': phi=1/(1+f*f)
     elif kind=='gauss': phi=np.exp(-f*f)
     elif kind=='power': phi=np.c_[f**2,f**3,f**4,f**5]
     elif kind=='quad+2ridge': phi=np.c_[np.tanh(f),np.tanh(w@p[5:9]+p[9])]
     else:
      r=np.sum((w-p[:4])**2,axis=1); phi=np.exp(-r*np.exp(p[4]));
      if kind=='radial+ridge':phi=np.c_[phi,np.logaddexp(0,w@p[5:9]+p[9])]
     return np.c_[D0,phi]
    def resid(p):
     a=feature(p)[train];return a@lin(a,t)-t
    seeds=[np.r_[d*1.5,.2],np.r_[vv[:,-1]*1.5,.2],np.r_[vv[:,0]*1.5,.2],np.r_[vv[:,-1]*3.,.8],np.r_[d*3.,.8]]
    if kind.startswith('radial'):seeds=[np.r_[d*c,p] for c,p in ((0,-1),(1,-1),(-1,-1),(0,-2),(1,0))]
    if kind in ('quad+2ridge','radial+ridge'):seeds=[np.r_[p,np.r_[v*1.8,.3]] for p,v in zip(seeds,[d,vv[:,-1],vv[:,0],d,vv[:,-1]])]
    if kind=='power':seeds=[p/np.linalg.norm(p[:4]) for p in seeds]
    best=min((least_squares(resid,p,bounds=(-8.,8.),max_nfev=75) for p in seeds),key=lambda o:np.linalg.norm(o.fun)); M=feature(best.x); err=np.mean(np.abs(M[test]@lin(M[train],t)-Y[test])); errs.append(err); params.append(best.x.round(2).tolist()); rms.append(np.std(best.fun));
   report.append((max(errs),errs[0],np.mean(errs),nm,kind,baseline,np.round(errs,4).tolist(),np.round(rms,4).tolist(),params[:2]))
print('best worst'); [print(t) for t in sorted(report)[:18]];print('best official');[print(t) for t in sorted(report,key=lambda t:t[1])[:8]]
