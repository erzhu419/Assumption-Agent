import numpy as np,pickle
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=np.c_[S['train']['y'],S['train']['traj']]; T=S['eval_times']; print('times',T,'SV',np.linalg.svd(Y[:,1:],compute_uv=False)); np.set_printoptions(precision=3,suppress=True)
def a(*z): return np.column_stack((np.ones(len(z[0])),*z))
def fit(D,v): return np.linalg.lstsq(D,v,rcond=None)[0]
def probe(D,v,rob):
 if rob:
  b=fit(D,v); r=least_squares(lambda p:(D@p.reshape(b.shape)-v).ravel(),b.ravel(),loss='soft_l1',f_scale=.008).x.reshape(b.shape); e=np.linalg.norm(D@r-v,axis=1); m=e<max(.04,4*np.median(e)); b=fit(D[m],v[m]);
 else: m=np.ones(len(v),bool); b=fit(D,v)
 e=np.linalg.norm(D@b-v,axis=1); print('n',m.sum(),'err',np.quantile(e,[0,.5,.8,.9,1]),'max in',max(e[m])); return b,m
for p in ['a','b','c']:
 P=S['panel_'+p]; u=P['u'];
 for key,v in P.items():
  if key=='u':continue
  print('panel',p,key)
  for name,D in [('lin',a(u)),('sq',a(u,u*u)),('cub',a(u,u*u,u**3)),('sin',a(u,np.sin(2.4*u))),('tanh',a(u,np.tanh(2.4*u))),('rational',a(u,u/(1+.6*u*u))),('diagcross',a(u,u*u,u*np.roll(u,2,1))),('odd',a(u,u**3)),('trig',a(u,np.sin(2.4*u),np.cos(2.4*u)))]:
   print(name,end=' '); b,m=probe(D,v,p=='a');
   if (name=='lin' and p=='a') or (name=='sq' and p!='a'):print(b)
