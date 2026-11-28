import pickle,numpy as np
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda F,y:np.linalg.lstsq(F,y,rcond=None)[0]
roll=lambda x:np.roll(x,-1,axis=1)
a=s['panel_a']; u,v=a['u'],a['v']; good=np.linalg.norm(v,axis=1)<5;u,v=u[good],v[good]
col=lambda z,j:np.c_[np.ones(len(z)),z[:,j],z[:,(j+1)%4],z[:,j]*z[:,(j+1)%4],z[:,j]**3]
C=np.stack([ls(col(u,j),v[:,j]) for j in range(4)],axis=1)
cal=lambda z:np.stack([col(z,j)@C[:,j] for j in range(4)],axis=1)
def inv(x):
 z=x.copy()
 for _ in range(30):
  J=(np.eye(4)[None,:,:]*(C[1]+C[3]*roll(z)+3*C[4]*z*z)[:,:,None]+np.roll(np.eye(4),1,axis=1)[None,:,:]*(C[2]+C[3]*z)[:,:,None]);d=np.linalg.solve(J, (cal(z)-x)[...,None])[...,0];z-=np.clip(d,-.5,.5)
 return z
x=s['train']['x']; z=inv(x);Y=np.c_[s['train']['y'],s['train']['traj']];print('cal',np.max(abs(cal(u)-v)),C,'inverse',np.max(abs(cal(z)-x)),z.min(0),z.max(0), 'test',inv(s['test']['x']).min(0),inv(s['test']['x']).max(0))
aff=lambda z:np.c_[np.ones(len(z)),z]
def auxiliary(k,j):
 p=s[k];u=p['u'];v=p[j];best=[]
 fs={'rat':lambda w,x:x/(1+w*x*x),'atan':lambda w,x:np.arctan(w*x),'tanh':lambda w,x:np.tanh(w*x),'cos':lambda w,x:np.cos(w*x),'sin':lambda w,x:np.sin(w*x),'sqrt':lambda w,x:x/np.sqrt(1+w*x*x),'sinh':lambda w,x:np.sinh(w*x)}
 for key,f in fs.items():
  for extra in ('none','lin','cyc','sq','lin+sin'):
   def feat(x,w,f=f,extra=extra):
    E={'none':lambda:np.empty((len(x),0)),'lin':lambda:x,'cyc':lambda:np.c_[x,x*roll(x)],'sq':lambda:np.c_[x,x*x],'lin+sin':lambda:np.c_[x,np.sin(w*x)]}[extra]();return aff(np.c_[f(w,x),E])
   def calc(w):
    A=feat(u,w);c=ls(A,v);e=A@c-v;h=np.sum(A*np.linalg.pinv(A).T,axis=1);return np.mean((e/np.maximum(1-h[:,None],.01))**2)**.5,c
   grid=np.linspace(.15,4,24);i=np.argmin([calc(w)[0] for w in grid]);r=minimize_scalar(lambda w:calc(w)[0],bounds=(grid[max(0,i-1)],grid[min(23,i+1)]),method='bounded');best.append((r.fun,key,extra,r.x,feat,calc(r.x)[1]))
 best.sort(key=lambda v:v[0]);print(k,j,[(round(e,8),f,t,round(w,5)) for e,f,t,w,_,_ in best[:5]]);e,k,t,w,f,c=best[0];return lambda x:f(x,w)@c
B=auxiliary('panel_b','v'); C1=auxiliary('panel_c','v1'); C2=auxiliary('panel_c','v2')
D={'x':x,'z':z,'A(x)':cal(x)}
for tag,h in list(D.items()):
 for nam,f in [('B',B),('C1',C1),('C2',C2)]:
  D[nam+'('+tag+')']=f(h)
for tag,h in list(D.items()):
 if tag in ('x','z','A(x)'):continue
 for nam,f in [('B',B),('C1',C1),('C2',C2)]:D[nam+'('+tag+')']=f(h)
def score(F):
 P=F[52:]@ls(F[:52],Y[:52]);return np.mean(abs(P-Y[52:]),axis=0)
for dg in (1,2,3):
 out=[]
 for tag,h in D.items():
  A=PolynomialFeatures(dg).fit_transform(h);E=score(A);out.append((E[0],E[1:].mean(),tag))
 print('deg',dg,'y',sorted(out)[:6],'T',sorted(out,key=lambda r:r[1])[:6]);
for tag in ['z','B(z)','C1(z)','C2(z)']:
 h=D[tag];print('corr',tag,np.round(np.corrcoef(np.c_[h,Y[:,[0,-1]]].T)[:4,4:],3))
print('z,Y first',np.round(np.c_[z,Y[:,[0,1,4,8]]][:30],4))
