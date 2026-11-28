import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize_scalar
Q=np.linalg.svd(tr[:52],full_matrices=False)[2][:3]; T=np.c_[Y[:,0],tr@Q.T]; mu=T[:52].mean(0); T-=mu; R=[]
def fit(n,F):
 F=(F-F[:52].mean(0))/np.maximum(F[:52].std(0),1e-7); G=F@F[:52].T
 for p in [.003,.10,1.]:
  E=cho_solve(cho_factor(G[:52]+p*np.eye(52)),np.eye(52)); A=E@T[:52]; r=(A/(E.diagonal()[:,None]-1/(52*p))); e=G[52:]@A-T[52:]; R.append((np.abs(r[:,0]).mean(),np.abs(r[:,1:]@Q).mean(),np.abs(e[:,0]).mean(),np.abs(e[:,1:]@Q).mean(),n,p))
P={d:PolynomialFeatures(d,include_bias=False).fit_transform(x) for d in [1,2,3]}
def cr(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(x),-1)
for tag in ['x','z','inv']:
 a=sets[tag]; opts={'a':a,'b':B(a),'c':C(a),'d':D(a),'e':C(B(a)),'f':D(B(a)),'g':C(D(a)),'h':D(C(a)),'i':B(D(a)),'j':B(C(a)),'k':C((a-wb[0])@np.linalg.inv(wb[1:])),'l':D((a-wb[0])@np.linalg.inv(wb[1:]))}
 for (ni,i),(nj,j) in it.combinations_with_replacement(opts.items(),2):
  base=np.c_[P[2],i,j]; bs=tag+ni+nj
  for nm,h in [('none',i),('allprod',cr(i,j)),('diag',i*j),('dot',(i*j).sum(1)[:,None]),('diag3',np.c_[i*i*j,i*j*j])]:fit(bs+nm,np.c_[base,h]);
  for k in [.22,1.25,2.4]:
   for nm,h in [('ratio',np.c_[i/(1+k*j*j),j/(1+k*i*i)]),('rat2',np.c_[i*i/(1+k*j*j),j*j/(1+k*i*i)]),('divprod',i*j/(1+k*(i*i+j*j))),('dotrat',np.c_[i/(1+k*np.sum(j*j,1)[:,None]),j/(1+k*np.sum(i*i,1)[:,None])]),('trig',np.c_[np.sin(k*(i+j)),np.cos(k*(i-j))]),('sqrt',np.sqrt(.01+(i+j)**2)),('saturation',np.c_[np.tanh(k*(i+j)),np.tanh(k*(i-j))])]:
    fit(bs+nm+str(k),np.c_[base,h]);fit(bs+'x'+nm+str(k),np.c_[base,h,cr(x,h)])
 # Allow both nonlinear branches to receive independently transformed arguments.
 for ni,i in opts.items():
  for nj,j in opts.items():
   for mode in range(4):
    h=[C(i),D(i)][mode//2]; g=[C(j),D(j)][mode%2]
    fit(tag+ni+nj+'nested'+str(mode),np.c_[P[2],h,g,cr(h,g)]);
    fit(tag+ni+nj+'nested_diag'+str(mode),np.c_[P[2],h,g,h*g]);
for k in range(5):
 f=(lambda a:a[0]+a[1]) if k==4 else (lambda a:a[k]);print('RESULT',k,sorted(R,key=f)[:6]);
