import pickle, runpy, numpy as np
from itertools import combinations
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; n=runpy.run_path(P+'node_00.py'); sink=runpy.run_path(P+'_frozen_nodes/node_01.py'); z=n['run_node']({'__input__':s})['z'][:72,:4]; y=s['train']['y']; t=s['eval_times']; T=s['train']['traj']; c=s['panel_c']; p=s['panel_b']
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(g,u,v):return g(z)@np.linalg.lstsq(g(u),v,rcond=None)[0]
B=fit(aff,p['u'],p['v']); C=n['_panel_c_v2'](c,z); U=np.cos(2.4*z)
pair=list(combinations(range(4),2)); cyc=[(0,1),(1,2),(2,3),(3,0)]
def prod(x,pairs=pair):return np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
for tag,g in [('aff',aff),('pair',lambda u:np.c_[aff(u),prod(u)]),('cube',lambda u:np.c_[aff(u),u**3]),('quadratic',lambda u:np.c_[aff(u),u*u,prod(u)]),('cyclic-cube',lambda u:np.c_[aff(u),prod(u,cyc),u**3]),('sin',lambda u:np.c_[aff(u),np.sin(2.4*u)])]:
 D=g(c['u']); e=(D@np.linalg.lstsq(D,c['v1'],rcond=None)[0]-c['v1'])/(1-np.diag(D@np.linalg.pinv(D)))[:,None]; print('C1',tag,np.sqrt(np.mean(e**2)),flush=True)
for count in range(2,6):
 def basis(r):return -np.expm1(-t[:,None]*r)
 def resid(r):
  h=basis(r); return (h@np.linalg.lstsq(h,T.T,rcond=1e-10)[0]-T.T).ravel()
 opt=least_squares(resid,np.geomspace(.35,4,count),bounds=(.005,30),max_nfev=100); er=resid(opt.x);print('modes',count,opt.x,np.mean(abs(er)),max(abs(er)),flush=True)
print('trajectory singular values',np.linalg.svd(T,compute_uv=False),flush=True)
V=fit(lambda u:np.c_[aff(u),prod(u,cyc),u**3],c['u'],c['v1'])
def expanded(q):
 L=[q,prod(q),q*q];
 for k in (.7,1.5,2.4,3.):
  for a,b in pair:
   x,w=q[:,a],q[:,b]; L.append(np.column_stack((np.sin(k*(x+w)),np.cos(k*(x+w)),np.sin(k*(x-w)),np.cos(k*(x-w)),np.sin(k*x*w),np.cos(k*x*w),1/(1+k*x*x*w*w),w*np.cos(k*x),x*np.cos(k*w))))
 return np.column_stack(L)
def inter(q,r):return (q[:,:,None]*r[:,None,:]).reshape(len(q),-1)
base={'raw':Z if (Z:=z) is not None else z,'C':np.c_[z,C],'U':np.c_[z,U],'B':np.c_[z,B,C],'V':np.c_[z,V,C],'C-int':np.c_[z,C,prod(C),inter(z,C)],'U-int':np.c_[z,U,prod(U),inter(z,U)],'expanded-C':np.c_[z,expanded(C)],'expanded-B':np.c_[z,C,expanded(B)],'expanded-V':np.c_[z,C,expanded(V)],'combo':np.c_[z,C,U,B,V,prod(C),prod(U),prod(B),prod(V),inter(z,C),inter(B,C),inter(V,C)],'B-first':np.c_[B,z,C],'C-first':np.c_[C,z,U],'V-first':np.c_[V,z,C]}
rg=np.random.default_rng(620); splits=[np.array_split(rg.permutation(len(y)),4) for _ in range(2)];
for tag,G in base.items():
 errors=[]
 for fold in splits[0]:
  a=np.setdiff1d(np.arange(len(y)),fold); h,m=sink['_modes'](t,T[a]); Y=np.zeros((len(y),3)); Y[:,0]=y;Y[a,1:]=m;K=sink['_covariances'](sink['_groups'](G),a); pred=np.column_stack([sink['_predict'](K,a,fold,v,.15) for v in Y.T]); errors.append([np.mean(abs(pred[:,0]-y[fold])),np.mean(abs(pred[:,1:]@h.T-T[fold]))]);
 print(tag,'mean',np.mean(errors,0).round(5),'worst fold',np.max(errors,0).round(5),flush=True)
