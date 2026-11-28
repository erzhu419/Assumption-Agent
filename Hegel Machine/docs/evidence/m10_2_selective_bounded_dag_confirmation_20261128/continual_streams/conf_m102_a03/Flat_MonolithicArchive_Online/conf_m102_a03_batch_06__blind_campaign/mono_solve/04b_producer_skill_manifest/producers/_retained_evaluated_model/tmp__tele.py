exec(open('/tmp/readout.py').read().split('for nam,a0')[0])
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
u=s['panel_a']['u']; v=s['panel_a']['v'];
def cm(u,c):
 n=np.roll(u,-1,axis=1);return c[:4]+c[4:8]*u+c[8]*n+c[9]*u*n+c[10]*u**3
print('cal residual',np.sort(np.linalg.norm(cm(u,cal)-v,axis=1)))
Y=np.c_[y,T]; splits=[(ix,val),*KFold(5,shuffle=True,random_state=41).split(z)];
def evalq(Q, regs=(.000001,.003,.03,.2)):
 out=[]
 for i,j in splits:
  Q0=Q-Q[i].mean(0); Q0/=Q0[i].std(0).clip(.00001); Q0=af(Q0); U,S,V=np.linalg.svd(Q0[i],full_matrices=False); out.append([np.mean(abs((Q0[j]@V.T)@((S/(S*S+r))[:,None]*(U.T@Y[i]))-Y[j]),0) for r in regs]);
 E=np.array(out); return [(round(e[0,0],5),round(e[1:,0].mean(),5),round(e[1:,0].max(),5),round(e[0,1:].mean(),5),r) for e,r in zip(E.transpose(1,0,2),regs)]
def warp(v,t,k):return np.sign(v)*np.maximum(abs(v)-t,0)**k
R=[]
for t in (0,.1,.16,.22,.28,.35,.42,.5):
 for k in (.75,1,1.25,1.5,1.75,2):
  g=warp(z,t,k)
  for nm,Q in [('g2',polys(g,2)),('gz2',polys(np.c_[g,z],2)),('hg2',polys(np.c_[g,h],2)),('hzg',np.c_[polys(h,2),(h[:,:,None]*g[:,None,:]).reshape(-1,16),z,g]),('hgdiag',np.c_[polys(h,2),g,h*g]),('wgdiag',np.c_[polys(w,2),g,w*g])]:
   for e in evalq(Q):R.append((*e,nm,t,k))
print('top gate',*sorted(R)[:14],sep='\n'); print('top cv',*sorted(R,key=lambda e:e[1])[:14],sep='\n'); print('top state',*sorted(R,key=lambda e:e[3])[:6],sep='\n')
# Profile a common dead-zone power of the calibrated inputs independently of panel B, then fit each readout.
for mode in ('quadratic','mixed','both'):
 def design(p,z):
  g=warp(z,*p); b0=af(polys(g,2));
  if mode=='mixed':b0=np.c_[b0,z,(g[:,:,None]*z[:,None,:]).reshape(-1,16)]
  if mode=='both':
   h=warp(z,.35,1.5); b0=np.c_[b0,polys(h,2),(g[:,:,None]*h[:,None,:]).reshape(-1,16)]
  return b0
 for jj in (slice(0,1),slice(1,None),slice(None)):
  out=[]
  for i,j in splits:
   def fun(p):
    Q=design(p,z[i]); A=Y[i,jj]; return (Q@fit(Q,A)-A).ravel()
   p=min([least_squares(fun,b,bounds=([0,.5],[.75,4]),max_nfev=100) for b in ([.22,1.25],[.35,1.5],[.5,2])],key=lambda o:o.cost)
   q=design(p.x,z); pr=q@fit(q[i],Y[i]); e=np.mean(abs(pr[j]-Y[j]),0); out.append((p.x,e[0],e[1:].mean(),np.mean(abs(p.fun))))
  print('profile',mode,jj,out)
