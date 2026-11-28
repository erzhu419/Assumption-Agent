import pickle,numpy as np,itertools,sys
from scipy.special import erf
from scipy.optimize import least_squares
sys.dont_write_bytecode=True;sys.path.insert(0,'/holdout_task');import node_00 as m
path='/holdout_task/_frozen_nodes/';f1={};f2={}
exec(open(path+'node_01.py').read(),f1);exec(open(path+'node_02.py').read(),f2)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=map(s['panel_b'].get,('u','v'));z=m._invert(s['train']['x'],m._fit_a(s['panel_a']));truth=np.c_[s['train']['y'],s['train']['traj']];n=len(z)
folds=[]
for seed in range(5):
 for te in np.array_split(np.random.default_rng(seed).permutation(n),5):
  tr=np.setdiff1d(np.arange(n),te);folds.append((tr,te,f1['run_node']({'__input__':{'train':{'y':truth[tr,0],'traj':truth[tr,1:]}}})))
def score(z,b):
 out=[]
 for tr,te,p in folds:
  o=f2['run_node']({'node_00':dict(z_train=z[tr],z_test=z[te],b_train=b[tr],b_test=b[te]),'node_01':p});e=np.abs(np.c_[o['y_hat'],o['state_pred']]-truth[te]);out.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(out).reshape(5,5,2).mean(1)
def report(name,z,b,res=0):
 a=score(z,b);print(name,round(res,6),a.mean(0).round(5),a.max(0).round(5),flush=True)
def derive(fn,init,lo,hi):
 def fit(q):
  D=np.c_[np.ones(len(u)),fn(u,q)];C=np.linalg.lstsq(D,v,rcond=None)[0];return C,D@C-v
 sol=min([least_squares(lambda q:fit(q)[1].ravel(),x,bounds=(lo,hi),max_nfev=250) for x in init],key=lambda o:o.cost);q=sol.x;c,e=fit(q);return fn(z,q),c[1:],np.mean(e**2)**.5,q
base=m._fold_b(s['panel_b'],z);report('base',z,base);report('z only',z,z)
for name,fn,init,lo,hi in [
 ('erf',lambda a,q:erf(q[0]*a)+q[1]*a,[[.6,-.6],[1.5,-1.55],[3,-2]],[.05,-10],[8,10]),
 ('exp',lambda a,q:a*np.exp(-(q[0]*a)**2)+q[1]*a,[[.7,-.8],[1,-.9],[2,-1]],[.05,-10],[8,10]),
 ('erf separate',lambda a,q:erf(a*q[:4])+a*q[4:8],[[1.5]*4+[-1.55]*4],[.05]*4+[-10]*4,[8]*4+[10]*4),
 ('erf mix',lambda a,q:erf(q[0]*a)+q[1]*a+q[2]*np.sin(2*a),[[1.5,-1.55,0],[.8,-.8,0]],[.05,-10,-10],[8,10,10]),
 ('erf cubic',lambda a,q:erf(q[0]*a)+q[1]*a+q[2]*a**3,[[1.5,-1.55,0],[.8,-.8,0]],[.05,-10,-10],[8,10,10]),
 ('tanh cubic',lambda a,q:np.tanh(q[0]*a)+q[1]*a+q[2]*a**3,[[1.5,-1.55,0],[.8,-.8,0]],[.05,-10,-10],[8,10,10]),
 ('rational cubic',lambda a,q:(a+q[1]*a**3)/(1+q[0]*a*a),[[1,-1],[2,-2]],[0,-10],[15,10]),
 ('rational quintic',lambda a,q:(a+q[1]*a**3+q[2]*a**5)/(1+q[0]*a*a),[[1,-1,0],[2,-2,0]],[0,-10,-10],[15,10,10]),
 ('quintic',lambda a,q:a+q[0]*a**3+q[1]*a**5,[[-2,1],[-5,3]],[-30,-30],[30,30])]:
 f,C,res,q=derive(fn,init,lo,hi);b=f@C;print('q',name,q.round(5),flush=True);report(name,z,b,res);report(name+' source',z,f,res)
 if name in ('erf','exp'):
  white=lambda x:(lambda a:a[0]*np.sqrt(len(x)))(np.linalg.svd(x-x.mean(0),full_matrices=False));report(name+' white b',z,white(b),res)
  for i in (.25,.5,1,2,4,8):
   g=lambda a:(a-a.mean(0))/a.std(0)
   for kind,w in [('b-only',np.c_[g(b),i*g(f)]),('z-fold',np.c_[g(z),i*g(f)])]:report(name+' '+kind+' '+str(i),w if kind=='z-fold' else z,w if kind=='b-only' else b,res)
