from compose import *
from scipy.special import erf

def calibrate(a,x):
 out=[]
 for u,v,z in zip(a['u'].T,a['v'].T,x.T):
  D=aff(u); b=ls(D,v)
  for _ in range(8):
   j=np.argsort(abs(D@b-v))[:int(.875*len(u))];b=ls(D[j],v[j])
  j=abs(D@b-v)<.002;b=ls(D[j],v[j]);out.append((z-b[0])/b[1])
 return np.array(out).T

def fit_panel(x,y, idx):
 res=[]
 for nm in (['erf','tanh','rat','sin','atan','asinh','inv3','inv5','softsign'] if idx!=2 else ['logcosh']):
  f=fs[nm]
  for mode in ['shared','shift','four','fourshift']:
   def D(x,p,mode=mode,f=f):
    g=p[0] if mode in ['shared','shift'] else p[:4]
    b=p[1] if mode=='shift' else p[4:8] if mode=='fourshift' else 0.
    return np.c_[aff(x),f(g*x+b)]
   def err(p):
    A=D(x,p);return (A@ls(A,y)-y).ravel()
   g=min(np.geomspace(.2,6,45),key=lambda g:np.linalg.norm(err(np.r_[np.repeat(g,4),np.zeros(4)] if mode.startswith('four') else [g,0] if mode=='shift' else [g])))
   m=4 if mode.startswith('four') else 1; nb=4 if mode=='fourshift' else 1 if mode=='shift' else 0
   o=least_squares(err,np.r_[np.repeat(g,m),np.zeros(nb)],bounds=(np.r_[np.repeat(.08,m),np.repeat(-1.5,nb)],np.r_[np.repeat(10.,m),np.repeat(1.5,nb)]),max_nfev=100)
   p=o.x; A=D(x,p); b=ls(A,y); J=np.c_[np.kron(A,np.eye(4)),o.jac]; h=np.sum(J*np.linalg.pinv(J).T,axis=1); score=np.std(o.fun/np.maximum(.05,1-h)); res.append((score,np.std(o.fun),nm,mode,p,b,lambda x,D=D,p=p,b=b:D(x,p)@b))
 return sorted(res,key=lambda v:v[0])

if __name__=='__main__':
 tr=s['train'];X=calibrate(s['panel_a'],tr['x']); T=tr['traj']; y=tr['y'];t=s['eval_times']; Y=np.c_[y,T]; allops=[]
 for i,(k,j) in enumerate([('panel_b','v'),('panel_c','v1'),('panel_c','v2')]):
  rr=fit_panel(s[k]['u'],s[k][j],i);allops.append(rr); print(k,j)
  for v in rr[:6]:print(v[:5]);
  print('best matrices',rr[0][5]);print('matrix ratio',rr[0][5][1:5]/rr[0][5][5:9]);
 def evaluate(D):
  A=D[:52];e=D@ls(A,Y[:52])-Y;h=np.sum(A*np.linalg.pinv(A).T,axis=1)[:,None];L=np.mean(abs(e[:52]/(1-h)),0);H=np.mean(abs(e[52:]),0);return np.r_[L[0],L[1:].mean(),H[0],H[1:].mean(),H[1:].max()]
 for b in allops[0][:3]:
  w=b[-1](X)
  for c in allops[1][:3]:
   q=c[-1](w); print('paths',b[2:4],c[2:4],evaluate(aff(q)),evaluate(poly(q,2)))
 for name,w in [('a0',allops[0][0][-1](X)),('a01',allops[1][0][-1](allops[0][0][-1](X)))]:
  r=[]
  for fn in ['id','tanh','atan','asinh','inv3','rat','sin','logcosh']:
   for g in ([1.] if fn=='id' else np.linspace(.5,6,23)):
    D=poly(fs[fn](g*w),2);r.append((evaluate(D),fn,g))
  print('y candidates',name,*sorted(r,key=lambda z:z[0][0])[:5],sep='\n')
