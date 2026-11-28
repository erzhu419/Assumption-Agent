import numpy as np,pickle,importlib.util
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; n=len(X)
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A=lambda x: np.column_stack([np.ones(len(x)),x]); mae=lambda x: np.mean(abs(x),axis=0)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
from scipy.optimize import least_squares
p=S['panel_c']; u=p['u']; D=A(u); V=p['v1']; W=p['v2']; b=np.linalg.lstsq(D,V,rcond=None)[0];
for label,fn,init,lo,hi in [
 ('tanh',lambda d,q: q[5]+q[6]*np.tanh(d@q[:5]), lambda j: np.r_[b[:,j]*1.25,0,1.15], [-np.inf]*6+[.1],[np.inf]*7),
 ('tanhlinear',lambda d,q: q[5]+q[6]*np.tanh(d@q[:5])+q[7]*(d@q[:5]),lambda j:np.r_[b[:,j]*1.25,0,1.,.1],[-np.inf]*6+[.1,-2],[np.inf]*7+[2]),
 ('rational',lambda d,q:q[5]+q[6]*(d@q[:5])/(1+.22*(d@q[:5])**2),lambda j:np.r_[b[:,j],0,1.],[-np.inf]*6+[.1],[np.inf]*7),
 ('atan',lambda d,q:q[5]+q[6]*np.arctan(d@q[:5]),lambda j:np.r_[b[:,j],0,1.],[-np.inf]*6+[.1],[np.inf]*7),
 ('sin',lambda d,q:q[5]+q[6]*np.sin(d@q[:5]),lambda j:np.r_[b[:,j],0,1.],[-np.inf]*6+[.1],[np.inf]*7)]:
 Q=[]; err=[]; cv=[]
 for j in range(4):
  for k in (24,18):
   opt=least_squares(lambda q:fn(D[:k],q)-V[:k,j],init(j),bounds=(lo,hi),max_nfev=2000); e=fn(D,opt.x)-V[:,j]
   if k==24: Q.append(opt.x);err.append(np.sqrt(np.mean(e**2)))
   else: cv.append(np.sqrt(np.mean(e[k:]**2)))
 print(label,'fit',err,'CV',cv,'coeff',np.array(Q))
def quad(g):return np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(4),2)])
for k,g in [('u',u),('v1',V),('u*v1',u*V)]:
 for key,E in [('q',np.c_[D,quad(g)]),('diag',np.c_[D,g*g]),('withv1',np.c_[D,V,g*g]),('withrat',np.c_[D,g*g,g/(1+.22*g*g)]),('withtanh',np.c_[D,g*g,np.tanh(1.25*g)])]:
  fit=np.linalg.lstsq(E[:18],W[:18],rcond=None)[0]; print(k,key,'CV',np.sqrt(np.mean((E[18:]@fit-W[18:])**2)))
E=np.c_[D,quad(u)]; Q=np.linalg.lstsq(E,W,rcond=None)[0]; print('v2 quad coeff',Q); print('fit residuals',E@Q-W)
