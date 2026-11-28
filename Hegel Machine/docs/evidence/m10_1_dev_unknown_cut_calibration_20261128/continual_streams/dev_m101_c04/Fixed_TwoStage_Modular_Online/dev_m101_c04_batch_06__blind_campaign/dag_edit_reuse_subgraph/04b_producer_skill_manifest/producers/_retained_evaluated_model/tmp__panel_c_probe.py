import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
frozen={}; exec(open('/holdout_task/_frozen_nodes/node_01.py').read(),frozen)
def affine(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
u,v=(s['panel_a'][k] for k in ('u','v'))
a=affine(u); m=np.ones(len(u),bool)
for _ in range(6):
 r=np.linalg.norm(a@ls(a[m],v[m])-v,axis=1);m=np.argsort(r)[:len(u)-3]
coef=[]
for j in range(4):
 k=(j+1)%4; D=np.c_[np.ones(len(u)),u[:,j],u[:,k],u[:,j]*u[:,k],u[:,j]**3]; c=ls(D[m],v[m,j]);coef.append(c);print('a channel',j,c.round(6),'rms',np.sqrt(np.mean((D[m]@c-v[m,j])**2)))
c=np.array(coef); b=ls(affine(s['panel_b']['u']),s['panel_b']['v'])
def A(x):
 n=np.roll(x,-1,axis=-1);return c[:,0]+c[:,1]*x+c[:,2]*n+c[:,3]*x*n+c[:,4]*x**3
def Ai(x):
 out=[]
 for row in x:
  q=least_squares(lambda z:A(z)-row,(row-c[:,0])/c[:,1],gtol=1e-12)
  out.append(q.x)
 return np.array(out)
def B(x):return affine(x)@b
def Bi(x):return np.linalg.solve(b[1:].T,(x-b[0]).T).T
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
trans={'raw':lambda x:x,'A':A,'Ai':Ai,'B':B,'Bi':Bi,'BAi':lambda x:B(Ai(x)),'BiAi':lambda x:Bi(Ai(x)),'AiBi':lambda x:Ai(Bi(x)),'AiB':lambda x:Ai(B(x)),'BA':lambda x:B(A(x)),'AB':lambda x:A(B(x))}
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; C={}
for key,fun in [('v1',np.tanh),('v2',np.cos)]:
 def basis(z,a): return np.c_[affine(z),fun(a*z)]
 def obj(a):
  D=basis(u,a); e=p[key]-D@ls(D,p[key]); return np.sum(e*e)
 a=minimize_scalar(obj,bounds=(.6,3.5),method='bounded').x
 w=ls(basis(u,a),p[key]); print(key,'frequency/RMS',a,(obj(a)/96)**.5,'coef',w.round(4)); C[key]=(lambda z,a=a,w=w,basis=basis,f=fun:np.c_[affine(z),f(a*z)]@w)
vals=[]
for name,F in trans.items():
 for h,H in [('id',lambda z:z),('C1',C['v1']),('C2',C['v2']),('C1+C2',lambda z:C['v1'](z)+C['v2'](z)),('C1-C2',lambda z:C['v1'](z)-C['v2'](z))]:
  z=H(F(x)); i=np.arange(54); j=np.arange(54,72)
  o=frozen['run_node']({'node_00':{'z_tr':z[i],'z_te':z[j],'y':y[i],'traj':t[i],'eval_times':s['eval_times'],'telemetry_vec':s['telemetry_vec']}})
  vals.append((np.mean(abs(o['y_hat']-y[j])),np.mean(abs(o['state_pred']-t[j])),name,h))
print('ranked',*sorted(vals,key=lambda v:max(v[:2]))[:15],sep='\n')
