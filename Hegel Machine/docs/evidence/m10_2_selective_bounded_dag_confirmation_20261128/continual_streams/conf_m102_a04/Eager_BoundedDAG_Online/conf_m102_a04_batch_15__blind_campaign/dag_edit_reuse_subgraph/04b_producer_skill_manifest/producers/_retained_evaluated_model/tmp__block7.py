import numpy as np,pickle
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def fit(f,y):
 b=np.linalg.lstsq(f,y,rcond=1e-8)[0]; e=y-f@b; h=np.sum(np.linalg.svd(f,full_matrices=False)[0]**2,axis=1); return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/np.maximum(1-h,1e-7)[:,None])**2)),b
for key,vkey in [('panel_b','v'),('panel_c','v2')]:
 u=s[key]['u']; y=s[key][vkey]; n=len(u); r=lambda k:np.roll(u,-k,axis=1); B=np.c_[np.ones(n),u]; print(key,'linear',fit(B,y)[:2],fit(B,y)[2]); qgrid=[.16,.32,.55,.8,1.25,1.6,2.15,2.7,3.3,4.3]; blocks={}
 def add(name,z):
  if z.ndim==1:z=z[:,None]
  blocks[name]=z
 for name,z in [('u',u),('sum1',u+r(1)),('dif1',u-r(1)),('sum2',(u+r(2))[:,:2]),('dif2',(u-r(2))[:,:2]),('prod1',u*r(1)),('prod2',(u*r(2))[:,:2]),('prod3',u*r(1)*r(2)),('mixedsum',u+.5*r(1)),('mixeddiff',u-.5*r(1)),('max',np.maximum(u,r(1))),('min',np.minimum(u,r(1)))]:
  add(name,z);add(name+'^2',z*z);add(name+'^3',z**3);add('abs'+name,abs(z));add('relu'+name,np.maximum(z,0))
  for q in qgrid:
   for nm,g in [('cos',np.cos(q*z)),('sin',np.sin(q*z)),('tanh',np.tanh(q*z)),('odd',z/(1+q*z*z)),('even',z*z/(1+q*z*z)),('soft',z/(1+q*abs(z)))]:add((nm,name,q),g)
 for k in [1,2,3]:
  add(('orig-cubic',k),u*u*r(k))
  for q in qgrid:
   for nm,g in [('sin-prod',np.sin(q*u)*r(k)),('cos-prod',np.cos(q*u)*r(k)),('cross-damp',u/(1+q*r(k)**2)),('cross-damp-prod',u*r(k)/(1+q*r(k)**2)),('prod-damp-sum',u*r(k)/(1+q*(u+r(k))**2)),('prod-damp-dif',u*r(k)/(1+q*(u-r(k))**2))]:add((nm,k,q),g)
 best=sorted((fit(np.c_[B,z],y)[1],fit(np.c_[B,z],y)[0],str(k),k) for k,z in blocks.items());print('best single',[(round(a,5),round(b,5),c) for a,b,c,k in best[:14]]);
 # Beam over multiple cyclic blocks, with and without a raw linear term.
 for base in [B,B[:,:1]]:
  beam=[(0,0,(),base)]; choices=list(blocks.items()); seen=set()
  for j in range(3):
   cand=[]
   for _,_,ks,f in beam:
    for k,z in choices:
     ss=tuple(sorted((*ks,str(k)))); ff=np.c_[f,z];
     if k in ks or ss in seen or ff.shape[1]>=n-3:continue
     seen.add(ss); m,cv,_=fit(ff,y);cand.append((cv,m,ss,ff))
   beam=sorted(cand,key=lambda z:z[:3])[:6]; print('beam',base.shape[1],j,[(round(a,6),round(b,6),c) for a,b,c,z in beam[:3]])
