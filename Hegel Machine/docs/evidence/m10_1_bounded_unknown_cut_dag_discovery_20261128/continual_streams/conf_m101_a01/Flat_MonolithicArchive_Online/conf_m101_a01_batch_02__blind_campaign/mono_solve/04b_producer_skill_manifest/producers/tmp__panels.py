import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
aug=lambda x:np.c_[np.ones(len(x)),x]
rms=lambda x:np.mean(x*x)**.5
u=s['panel_a']['u'];v=s['panel_a']['v'];m=(abs(v-u)<1.5)
ab=np.array([ls(aug(u[m[:,j],j:j+1]),v[m[:,j],j]) for j in range(4)])
b=s['panel_b']; B=ls(aug(b['u']),b['v'])
c=s['panel_c']; u=c['u']; Z=c['v1']; W=c['v2']; I=np.arange(len(u));
def fitop(out, feats, p0, ix):
 def err(p):
  a=feats(u[ix],p);return (a@ls(a,out[ix])-out[ix]).ravel()
 p=least_squares(err,p0,max_nfev=100).x if len(p0) else np.array([])
 return p,ls(feats(u[ix],p),out[ix]),rms(err(p))
ops=[]
for name,f,p0 in [('softpow',lambda x,p:x/(1+np.exp(p[0])*abs(x)**np.exp(p[1])),[0,0]),('softsign',lambda x,p:x/(1+np.exp(p[0])*abs(x)),[0]),('rational',lambda x,p:x/(1+np.exp(p[0])*x*x),[0]),('sqrt',lambda x,p:x/np.sqrt(1+np.exp(p[0])*x*x),[0]),('atan',lambda x,p:np.arctan(np.exp(p[0])*x),[0]),('tanh',lambda x,p:np.tanh(np.exp(p[0])*x),[0]),('tele-soft',lambda x,p:1.7*x/(1+1.7*abs(x)),[])]:
 for skip in [False,True]:
  g=lambda x,p,f=f,skip=skip:aug(np.c_[f(x,p),x] if skip else f(x,p))
  p,a,e=fitop(Z,g,p0,I[:18]); pe=rms(g(u[18:],p)@a-Z[18:]); p,a,e=fitop(Z,g,p0,I);ops.append((pe,name+str(skip),g,p,a));print('C1',name,skip,np.round(p,5),round(e,6),round(pe,6))
_,n,g,p,a=min(ops,key=lambda v:v[0]); C1=lambda x,g=g,p=p,a=a:g(x,p)@a
ops=[]
for name,g,p0 in [('diag',lambda x,p:aug(np.c_[x,x*x]),[]),('diag+cub',lambda x,p:aug(np.c_[x,x*x,x**3]),[]),('diag+quart',lambda x,p:aug(np.c_[x,x*x,x**4]),[]),('diag+abs',lambda x,p:aug(np.c_[x,x*x,x*abs(x)]),[]),('rational2',lambda x,p:aug(np.c_[x,x*x/(1+np.exp(p[0])*x*x)]),[-3]),('rationalboth',lambda x,p:aug(np.c_[x/(1+np.exp(p[0])*x*x),x*x/(1+np.exp(p[1])*x*x)]),[-3,-3]),('tanhboth',lambda x,p:aug(np.c_[np.tanh(np.exp(p[0])*x),np.tanh(np.exp(p[1])*x*x)]),[-2,-2]),('allquad',lambda x,p:PolynomialFeatures(2).fit_transform(x),[])]:
 p,a,e=fitop(W,g,p0,I[:18]); pe=rms(g(u[18:],p)@a-W[18:]);p,a,e=fitop(W,g,p0,I);ops.append((pe,name,g,p,a));print('C2',name,np.round(p,5),round(e,6),round(pe,6))
_,n,g,p,a=min(ops,key=lambda v:v[0]); C2=lambda x,g=g,p=p,a=a:g(x,p)@a
X=s['train']['x']; Y=s['train']['y']; T=s['train']['traj'];
def routes(x):
 for name,d in [('X',x),('A',x*ab[:,1]+ab[:,0]),('Ai',(x-ab[:,0])/ab[:,1])]:
  for n,z in [(name,d),(name+'B',aug(d)@B),(name+'Bi',(d-B[0])@np.linalg.inv(B[1:]))]:yield n,z
