import pickle,numpy as np,sys
from itertools import combinations
sys.path.insert(0,'/holdout_task')
import node_00 as m
from _frozen_nodes import node_03 as sink, node_01 as basis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u'];v=p['v1']; n=len(u); C=lambda x:m._operator(p,'v2',x,False); B=lambda x:m._operator(s['panel_b'],'v',x,True)
x=m._calibrate(s['panel_a'],s['train']['x']); raw=s['train']['x']; b=B(x)
def score(xx,cc):
 D=basis._design(xx,cc); a,z=sink._features(xx,cc,D); out=[]
 for tr,va in splits:
  yp=sink._scalar(a[tr],z[tr],y[tr],a[va],z[va]); modes=np.linalg.svd(t[tr],full_matrices=False)[2][:2]; tp=D[va]@np.linalg.lstsq(D[tr],t[tr]@modes.T,rcond=None)[0]@modes; out.append((np.mean(abs(yp-y[va])),np.mean(abs(tp-t[va]))))
 return np.array(out)
from itertools import combinations_with_replacement
from scipy.special import erf
def d2(z): return np.column_stack((np.ones(len(z)),z,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)]))
w=np.linalg.lstsq(d2(u),v,rcond=None)[0]; print('v1 weights',np.round(w,3)); Q=lambda z:d2(z)@w
print('A inverse calibration error',np.quantile(abs(m._calibrate(s['panel_a'],s['panel_a']['v'])-s['panel_a']['u']),[.5,.9,.93]))
y=s['train']['y'];t=s['train']['traj']; splits=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]
ops={'B':B,'Q':Q,'C':C}; base={'x':x,'raw':raw}; D=dict(base)
for i in range(3):
 for k,z in list(D.items()):
  if len(k.split('/'))!=i+1:continue
  for op,f in ops.items():D[k+'/'+op]=f(z)
from scipy.linalg import lstsq
rows=[]
for xn,xx in list(D.items()):
 if xn.count('/')>1:continue
 for cn,cc in D.items():
  if np.max(abs(cc))>50:continue
  ds=basis._design(xx,cc); sc=[]
  for tr,va in splits:
   tp=ds[va]@np.linalg.lstsq(ds[tr],t[tr],rcond=None)[0];sc.append(np.mean(abs(tp-t[va])))
  rows.append((max(sc),xn,cn,np.round(sc,5)))
print('best compositions');
for _,xn,cn,sc in sorted(rows,key=lambda r:r[0])[:18]: print(xn,cn,sc,np.round(score(D[xn],D[cn]),5))
for a in ['x','x/B','x/Q','x/C']:
 for b0,c0 in [('x/B/Q','x/B/C'),('x/Q','x/C'),('x/Q/B','x/C/B'),('x/Q','x/B'),('x/C','x/B')]:
  xx=D[a]
  for alpha in [.13,.22,.334,.5,.666,.78,.87,1,1.25,2.15,-.22,-1]:
   cc=D[b0]+alpha*D[c0]; ds=basis._design(xx,cc); sc=[np.mean(abs(ds[va]@np.linalg.lstsq(ds[tr],t[tr],rcond=None)[0]-t[va])) for tr,va in splits]; rows.append((max(sc),a,(b0,c0,alpha),np.round(sc,5)))
print('best including mixtures');print(*sorted(rows,key=lambda r:r[0])[:18],sep='\n')
