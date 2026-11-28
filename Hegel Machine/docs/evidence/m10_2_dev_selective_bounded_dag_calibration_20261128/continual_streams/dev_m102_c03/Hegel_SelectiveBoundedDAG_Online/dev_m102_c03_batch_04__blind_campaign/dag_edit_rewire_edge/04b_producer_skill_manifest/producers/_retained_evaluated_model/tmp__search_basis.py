exec(open('/tmp/panel_probe.py').read().split("print('panel1")[0])
from scipy.optimize import least_squares
from itertools import combinations,product
# Fit only the designated training rows. Report target-scale residuals.
R=np.column_stack([Y[:,0],Y[:,1:]@q.T]); U=np.linalg.svd(Y[:n]-Y[:n].mean(0),full_matrices=False); print('target sv',U[1]);
for k,p in [('B',s['panel_b']),('C',s['panel_c'])]:
 u=p['u']; print(k)
 for key,v in p.items():
  if key=='u':continue
  for name,F in [('aff',aff(u)),('tanh',aff(np.tanh(u))),('poly',poly(u)),('sin',aff(np.sin(u))),('cos',aff(np.cos(2.4*u)))]:
   print(key,name,np.mean((F@ls(F,v)-v)**2)**.5,ls(F,v).round(3) if k=='B' and name=='aff' else '')
p=s['panel_a']; u=p['u'];v=p['v'];D=aff(u); keep=np.linalg.norm(D@ls(D,v)-v,axis=1)<3; cal=np.array([ls(D[keep][:,[0,j+1]],v[keep,j]) for j in range(4)]).T;Z=(x-cal[0])/cal[1];
def calc(F,reg=0):
 if reg:
  A=F-F[:n].mean(0); A=A/np.maximum(A[:n].std(0),1e-7); pr=A@A[:n].T@np.linalg.solve(A[:n]@A[:n].T+reg*np.eye(n),R[:n]-R[:n].mean(0))+R[:n].mean(0)
 else:pr=F@ls(F[:n],R[:n]);
 return np.r_[abs(pr[n:,0]-R[n:,0]).mean(),abs(pr[n:,1:]-R[n:,1:]).mean(),abs(pr[:n]-R[:n]).mean()]
def quad(h):return poly(h)
def cf(h):return np.column_stack([np.ones(len(h)),h[:,0]*h[:,1],h[:,0]*h[:,3],h[:,1]*h[:,2],h[:,2]*h[:,3]])
p=s['panel_c']; W=ls(aff(np.cos(2.4*p['u'])),p['v2']); W1=ls(cf(p['u']),p['v1']); p=s['panel_b'];u=p['u'];v=p['v']; B=ls(aff(u),v);
Hs={'cal':Z,'raw':x,'B':aff(Z)@B,'Bi':(Z-B[0])@np.linalg.inv(B[1:])}; C=np.cos(2.4*Z); Q=cf(Z); V2=aff(C)@W;V1=Q@W1; print('ranges',Z.min(0),Z.max(0)); print('C matrices',W.round(3),W1.round(3));
rec=[]
def add(key,F,regs=(0,)):
 for r in regs: rec.append((key,r,*calc(F,r)))
for name,h in Hs.items():
 u,v=h[:,0],h[:,1]; pair=np.column_stack([h[:,i]+sg*h[:,j] for i,j in combinations(range(4),2) for sg in (1,-1)]); evenbase=np.column_stack([np.ones(len(h)),*[np.prod(h[:,j],axis=1) for j in combinations(range(4),2)]]); E=np.column_stack([np.ones(len(h)),h*h,cf(h)[:,1:]])
 for w in np.arange(.5,5.01,.1):
  c=np.cos(w*h); cp=np.cos(w*pair); si=np.sin(w*h); ca=np.cos(w*h)-1
  for k,F in [('cc',np.column_stack([quad(h),c,quad(c)])),('crossC',np.column_stack([E,c,cf(c)])),('sincross',np.column_stack([E,c,cf(si)])),('pCOS',np.column_stack([E,c,cp])),('pCOSred',np.column_stack([cf(h),c,cp])),('sc',np.column_stack([cf(h),c,cf(si),cf(c)])),('warp',quad(h+1.25*ca)),('warp2',quad(h+1.25*np.expm1(-.2962*h*h)))]:add((name,k,round(w,2)),F,(0,.3,3))
for title,v in [('v1',V1),('v2',V2),('both',V1+V2),('C',C-1),('Q',V1-W1[0])]:
 for f in [-1.25,-1,-.22,.22,1,1.25]:
  g=Z+f*v
  add((title,f,'warp'),quad(g),(0,.3,3));add((title,f,'warpc'),np.column_stack([quad(g),np.cos(2.4*g)]),(0,.3,3));
for k,F in [('Vpoly',poly(np.column_stack([V1,V2])) if False else np.column_stack([quad(V1),quad(V2),np.einsum('ni,nj->nij',V1,V2).reshape(len(x),-1)])),('CPoly',np.column_stack([cf(Z),poly(C,3),np.einsum('ni,nj->nij',Q[:,1:],C).reshape(len(x),-1)])),('radial',np.column_stack([quad(Z),*(np.cos(w*Z) for w in (1.,2.4,3.6,4.8)),*(np.exp(-a*Z**2) for a in (.3,1,2))]))]:add(k,F,(0,.01,.3,3,10))
print('best',*sorted(rec,key=lambda a:max(a[2:4]))[:25],sep='\n');print('best training',*sorted(rec,key=lambda a:a[-1])[:12],sep='\n');
