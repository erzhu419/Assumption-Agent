import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(x)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(z,d):return np.column_stack([np.prod(z[:,i],axis=1) if i else np.ones(len(z)) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; A=aff(u); g=np.ones(len(u),bool)
for i in range(4):
 c=fit(A[g],v[g]); e=np.linalg.norm(v-A@c,axis=1);g=e<4*np.median(e)
a=(x-c[0])/np.diag(c[1:]); b=s['panel_b']; B=fit(aff(b['u']),b['v']); z=aff(a)@B
np.set_printoptions(precision=4,suppress=True)
def metrics(D,target=y):
 c=fit(D[:52],target[:52]); e=D@c-target
 return np.mean(abs(e[:52])),np.mean(abs(e[52:])),np.max(abs(e[52:])),c
from scipy.optimize import least_squares
from itertools import combinations
rng=np.random.default_rng(813); hold=[np.arange(52,72),np.arange(56,72),*np.array_split(rng.permutation(n),5)]; splits=[(np.arange(len(y)-len(j)),j) if k<2 else (np.setdiff1d(np.arange(n),j),j) for k,j in enumerate(hold)]
w=z; h=w.T; p=h[0]*h[2]; q=h[1]*h[3]; lin=aff(w); pairs=np.c_[lin,p,q]; triples=np.column_stack([np.prod(w[:,ij],axis=1) for ij in combinations(range(4),3)])
print('B fit',B,'rms',np.std(aff(b['u'])@B-b['v'])); print('times',s['eval_times'],'summary',s['train']['summary'])
sgn=np.array([-1.,-1.,1.,1.]);
for tag,base in [('pair',pairs),('quad',poly(w,2))]:
 for label,source,initial in [('linear',w,[sgn,k*sgn if False else .4*sgn, np.array([1.,1,1,1])]),('triples',triples,[sgn,.4*sgn,np.ones(4)]),('pair',np.c_[p,q],[np.ones(2),np.array([1.,-1.])])]:
  for fun in ['sin','tanh','rat','cos','prodgate','divgate','trigprod']:
   def des(a):
    h=source@a[:-1]+a[-1]
    f={'sin':lambda:np.sin(h),'tanh':lambda:np.tanh(h),'cos':lambda:np.cos(h),'rat':lambda:h/(1+h*h),'prodgate':lambda:np.c_[p*np.tanh(h),q*np.tanh(h)],'divgate':lambda:np.c_[p/(1+h*h),q/(1+h*h)],'trigprod':lambda:np.c_[p*np.sin(h),q*np.sin(h)]}[fun]()
    return np.c_[base,f]
   out=[]; pp=[]
   for i,j in splits:
    def res(a):
     D=des(a)[i]; return D@fit(D,y[i])-y[i]
    opts=[least_squares(res,np.r_[a,.1],bounds=(-4,4),max_nfev=140) for a in initial]; r=min(opts,key=lambda r:r.cost); D=des(r.x); out.append(np.mean(abs(D[j]@fit(D[i],y[i])-y[j]))); pp.append(r.x)
   if min(out[:2])<.05 or np.mean(out[2:])<.042:print(tag,label,fun,np.round([out[0],out[1],np.mean(out[2:]),max(out[2:])],5),np.round(pp[0],3),flush=True)
