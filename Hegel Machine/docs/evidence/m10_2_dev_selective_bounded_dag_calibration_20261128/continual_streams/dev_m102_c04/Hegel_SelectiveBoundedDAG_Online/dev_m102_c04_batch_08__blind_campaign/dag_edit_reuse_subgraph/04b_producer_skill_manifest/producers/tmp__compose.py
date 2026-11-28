import pickle,runpy,numpy as np
from itertools import product
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; g=runpy.run_path('/holdout_task/node_00.py'); L=g['_ls']; aff=g['_affine']; a,A=g['_cal'](s['panel_a']); b,B=g['_cal'](s['panel_b']); Ai=np.linalg.inv(A); Bi=np.linalg.inv(B); x=s['train']['x']; k,wp,wr=g['_ops'](s['panel_c']); Q=g['_quad']; y=s['train']['y']; tr=s['train']['traj']; U,S,V=np.linalg.svd(tr[:52],full_matrices=False); T=np.c_[y,tr@V[:3].T]; scale=T[:52].std(0); T=T/scale;n=52
P=lambda z:aff(Q(z))@wp; R=lambda z:aff(np.cos(k*z))@wr
q=(x-a)@Ai; z={'x':x,'q':q,'Bq':q@B+b,'Iq':(q-b)@Bi,'Ax':x@A+a,'Bx':x@B+b,'Ix':(x-b)@Bi,'IBAx':((x-b)@Bi-a)@Ai,'Br':q@B,'Ir':q@Bi,'Bt':q@B.T,'Rr':q@B/.88,'Rt':q@B.T/.88,'Bcenter':(q-b)@B}
for j,h in product([.22,1.25,-.22,-1.25],[q@B,q@Bi,q@B+b,(q-b)@Bi]): z[f'm{j}_{len(z)}']=q+j*h
F={}; G={}
for name,v in z.items():
 F[name]=P(v);G[name]=R(v)
 for op,w in [('P',P(v)),('R',R(v))]:
  for tag,e in [('i',w),('B',w@B+b),('I',(w-b)@Bi),('p',v+.22*w),('m',v-.22*w),('s',v+1.25*w)]:
   F[name+op+tag]=P(e);G[name+op+tag]=R(e)
def check(h):
 f=aff(h); w=L(f[:n],T[:n]); e=np.abs(f@w-T); return np.r_[e[:n,:3].mean(),e[n:,0].mean()*scale[0],(e[n:,1:3]*scale[1:3]).mean()]
rank=[]
# First screen main effects and then interactions on strongest pairs.
for i,p in F.items():
 for j,r in G.items():
  v=check(np.c_[q,p,r,p*r]); rank.append((v[0],v[1],v[2],i,j))
for ix in range(3):print('screen',ix,*sorted(rank,key=lambda r:r[ix])[:7],sep='\n')
short=set((r[3],r[4]) for ix in range(3) for r in sorted(rank,key=lambda r:r[ix])[:70]); short.update(product(z,z)); best=[]
for i,j in short:
 p,r=F[i],G[j]; mul=p*r; dot=mul.sum(1); norm=(r*r).sum(1); cross=(p[:,:,None]*r[:,None,:]).reshape(-1,16)
 for nm,h in {'dot':dot,'mix':mul,'full':cross,'rat':np.c_[p/(1+.22*r*r),r/(1+.22*p*p)],'rat2':np.c_[p/(1+1.25*r*r),r/(1+1.25*p*p)],'ratnorm':np.c_[p/(1+.22*norm[:,None]),dot/(1+.22*norm)],'trig':np.c_[np.sin(1.25*(p+r)),np.cos(1.25*(p-r))],'prod3':np.c_[p*r*q,p*q,r*q],'squares':np.c_[p*p,r*r,(p*r).sum(1)],'gate':np.c_[p*np.tanh(1.25*r),p*np.tanh(.22*r)],'sumgate':np.c_[dot/(1+.22*dot**2),np.tanh(1.25*dot)],'cosq':np.c_[mul,np.cos(1.25*q),np.sin(1.25*q)]}.items():
  e=check(np.c_[q,p,r,h]);best.append((*e,i,j,nm))
for ix in range(3): print('extended',ix,*sorted(best,key=lambda r:r[ix])[:10],sep='\n')
print('target correlations',np.corrcoef(T.T));print('target extrema',T.min(0),T.max(0))
