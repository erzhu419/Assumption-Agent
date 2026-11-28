import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def poly(u,d=2):return np.column_stack([np.ones(len(u))]+[u[:,i].prod(1) for j in range(1,d+1) for i in combinations_with_replacement(range(u.shape[1]),j)])
def cal(u):return np.c_[poly(u),u**3]
p=s['panel_a']; pa=aff(p['u']); rows=np.linalg.norm(pa@fit(pa,p['v'])-p['v'],axis=1)<5; A=fit(cal(p['u'][rows]),p['v'][rows]); A[abs(A)<.0005]=0
print('A coefficients',A, 'resid',abs(cal(p['u'][rows])@A-p['v'][rows]).max())
def inverse(x):return np.array([least_squares(lambda q:(cal(q[None])@A)[0]-v,v).x for v in x])
F={'raw':x,'inv':inverse(x),'for':cal(x)@A}; print('raw std',x.std(0),'inv std',F['inv'].std(0))
func={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'cos':np.cos,'swish':lambda z:z*expit(z),'rat':lambda z:z/(1+z*z),'logcosh':lambda z:np.logaddexp(z,-z),'softplus':lambda z:np.logaddexp(z,0)}
def panel_fit(p,k,f,typ):
 u=p['u']; v=p[k]; a=aff(u); base=[a,np.ones((len(u),1))][typ]
 def compute(w):
  P=np.c_[base,f(w*u)]; return P,fit(P,v)
 def loss(w):
  P,b=compute(w); return np.mean((P@b-v)**2)
 g=np.linspace(.05,5,100); z=min(g,key=loss); o=minimize_scalar(loss,bounds=(max(.01,z-.07),z+.07),method='bounded'); P,b=compute(o.x); return o.fun**.5,o.x,b
D={}
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 d=[]
 for n,f in func.items():
  for typ in range(2):
   e,w,b=panel_fit(s[p],k,f,typ); d.append((e,n,typ,w,b))
 d.sort(key=lambda z:z[0]); print(p,k,[(round(e,6),n,j,round(w,5)) for e,n,j,w,_ in d[:8]]); print(d[0][-1]); D[k]=d[0]
def op(k,u):
 e,n,j,w,b=D[k]; base=[aff(u),np.ones((len(u),1))][j]; return np.c_[base,func[n](w*u)]@b
def report(d):
 a=d[:52]; b=d[52:]; scores=[]
 for h in [y,T]:
  c=fit(a,h[:52]); scores.extend([np.mean(abs(a@c-h[:52])),np.mean(abs(b@c-h[52:]))]);
 return np.round(scores,4)
for n,z in F.items():
 v,b,c=[op(k,z) for k in ('v','v1','v2')]; h=[('orig',z),('B',v),('C1',b),('C2',c),('C1+C2',b+c),('C1-C2',b-c),('B+C1',v+b),('B+C2',v+c),('C2(C1)',op('v2',b)),('B(C1)',op('v',b)),('B+C',v+b+c),('all',np.c_[z,v,b,c])]; print(n)
 for k,u in h: print(k,report(aff(u)),report(poly(u)) if u.shape[1]<5 else '',report(aff(np.c_[z,u])) if k!='orig' else '')
 for p,q in [(z,v),(z,b),(z,c),(b,c),(z,np.c_[v,b,c]),(z,np.c_[b,c])]:
  print('cross',report(aff(np.c_[p,q,(p[:,:,None]*q[:,None,:]).reshape(len(x),-1)])))
