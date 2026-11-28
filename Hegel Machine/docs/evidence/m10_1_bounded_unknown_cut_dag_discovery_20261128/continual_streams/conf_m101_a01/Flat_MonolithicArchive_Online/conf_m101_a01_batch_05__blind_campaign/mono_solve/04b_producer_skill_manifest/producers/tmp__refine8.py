from struct7 import *
from scipy.spatial.distance import cdist
I=np.eye(72)
print('time/tel/summary',s['eval_times'],s['telemetry_vec'],t['summary']);print('ranges y',y.min(),y.max())
def errors(K,reg,target=y):
 return np.array([np.mean(abs(K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*I[:len(tr),:len(tr)],target[tr])-target[te])) for tr,te in splits])
def ker(a,l):
 a=a/np.mean(a.std(0));return (1+a@a.T/a.shape[1])**2,np.exp(-cdist(a,a,'sqeuclidean')/(2*a.shape[1]*l*l))
results=[]
for base,aa in [('x',x),('d',d),('f',x*gain+bias)]:
 for off in [0,1]:
  b0=aa@B[1:]+off*B[0]; c=w(b0)-w(np.zeros((1,4))); h=aa+c@B[1:];
  for mix,l,reg in product([.65,.85,1,1.2,1.5],[.65,.9,1.2,1.6,2.2],[1e-6,3e-5,.0003,.002]):
   a=aa+mix*c@B[1:]; p,r=ker(a,l);_,r0=ker(aa,l);_,rc=ker(c,l)
   for name,K in [('h',p+r),('h+raw',p+.8*r+.2*r0),('h+c',p+.8*r+.2*rc)]:
    e=errors(K,reg);results.append((e.mean(),e.max(),*e[:2],base,off,mix,l,reg,name))
  for name,a in [('branch',c),('sum',h),('raw,branch',np.c_[aa,c])]:
   for k in [1,2,3]:
    a1=aug(old._poly(a,range(1,k+1)));coef=ls(a1,y);print('train',base,off,name,k,np.mean(abs(a1@coef-y)))
print('mean',*sorted(results)[:12],sep='\n');print('worst',*sorted(results,key=lambda a:a[1])[:12],sep='\n')
