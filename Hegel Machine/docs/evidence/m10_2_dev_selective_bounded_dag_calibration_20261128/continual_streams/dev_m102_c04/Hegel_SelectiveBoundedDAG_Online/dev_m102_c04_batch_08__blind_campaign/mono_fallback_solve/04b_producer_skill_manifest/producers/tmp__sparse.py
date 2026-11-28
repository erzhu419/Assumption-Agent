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
U,S,V=np.linalg.svd(tr[:52],full_matrices=False); T=np.c_[Y[:,0],tr@V[:3].T]; sc=T[:52].std(0); mu=T[:52].mean(0); T=(T-mu)/sc

def score(p):
 r=(p-T[52:])*sc; return np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]@V[:3]))

def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(x),-1)
for coord in ['inv','x','z','Binv','Bz']:
 q=sets[coord]; p=PolynomialFeatures(4,include_bias=False).fit_transform(q); f=np.cos(2.4*q); j=np.sin(2.4*q); qq=p[:,4:14];
 blocks={'poly':p,'cos':f,'sin':j,'xcos':cross(q,f),'xsin':cross(q,j),'qqcos':cross(qq,f),'coscos':PolynomialFeatures(2,include_bias=False).fit_transform(f)[:,4:],'sin2':np.sin(4.8*q),'cos2':np.cos(4.8*q),'frac':q/(1+.22*q*q)}
 for level,names in enumerate([['poly'],['poly','cos','xcos'],['poly','cos','sin','xcos','xsin'],list(blocks),list(blocks)+['branches']]):
  if 'branches' in names:
   for nm,r in [('C',C(q)),('D',D(q)),('DB',D(B(q))),('DC',D(C(q))),('CD',C(D(q)))]:
    blocks[nm]=np.c_[r,cross(q,r),r/(1+.22*r*r)];
   names=list(blocks)
  fs=[]; tags=[]
  for nm in names:
   f=blocks[nm]; fs.append(f); tags.extend([nm+str(i) for i in range(f.shape[1])]);
  F=np.column_stack(fs); F-=F[:52].mean(0);F/=np.maximum(F[:52].std(0),1e-8); Ft=F[:52]; test=F[52:];
  for noise in [.003,.015,.06]:
   for mode in [0,1]:
    vv=np.full(F.shape[1],1/F.shape[1]); eye=np.eye(52)*noise**2
    for k in range(220):
     W=Ft*vv; c=cho_factor(W@Ft.T+eye); a=cho_solve(c,T[:52]); co=W.T@a; v=np.sum(cho_solve(c,Ft)*Ft,axis=0); update=np.mean(co**2,axis=1)/np.maximum(v*vv,1e-18) if mode==0 else np.sqrt(np.mean(co**2,axis=1)/np.maximum(v,1e-18)); vv=np.clip(.35*vv+.65*update,1e-13,10)
    meas=score(test@co); resid=np.std(Ft@co-T[:52],axis=0); 
    if max(meas)<.18 or noise==.015 and mode==0: print('sparse',coord,level,noise,mode,np.round(meas,4),resid.round(4),[(tags[k],round(vv[k],3)) for k in np.argsort(vv)[-8:][::-1]],flush=True)
