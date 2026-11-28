import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x); t=s['eval_times']; T=tr['traj']; Y=np.c_[tr['y'],T]; np.set_printoptions(precision=4,suppress=True)
def fit(f,v): return np.linalg.lstsq(f,v,rcond=1e-10)[0]
def ones(u): return np.ones(len(u))
def quad(u): return np.c_[ones(u),u,*[u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)]]
p=s['panel_a'];u=p['u'];v=p['v'];h=np.c_[ones(u),u];g=np.linalg.norm(v-h@fit(h,v),axis=1)<3;u=u[g];v=v[g];w=np.roll(u,-1,axis=1)
for j in range(4):
 a=u[:,j]; b=w[:,j]; y=v[:,j]; mats=np.c_[ones(u),a,b,a*b,a*a,b*b,a**3,b**3,a*a*b,a*b*b,np.sin(2*a),np.sin(2*b)];print('A',j,fit(mats[:,:10],y),np.std(y-mats[:,:4]@fit(mats[:,:4],y)),np.std(y-mats[:,:10]@fit(mats[:,:10],y)))
def cal(u):
 w=np.roll(u,-1,axis=1);return np.stack([np.ones_like(u),u,w,u*w],axis=2)
c=np.stack([fit(cal(u)[:,j],v[:,j]) for j in range(4)]);print('cal',c)
def inv(x,c):
 z=(x-c[:,0])/c[:,1]
 for _ in range(30):z=(x-c[:,0]-c[:,2]*np.roll(z,-1,axis=1))/(c[:,1]+c[:,3]*np.roll(z,-1,axis=1))
 return z
z=inv(x,c); xs=np.r_[x,s['test']['x']]; zz=inv(xs,c);print('ranges',xs.min(0),xs.max(0),zz.min(0),zz.max(0), 'invert error',np.max(abs((cal(zz)*c).sum(2)-xs)))
p=s['panel_b'];b=fit(np.c_[ones(p['u']),p['u']],p['v']);p=s['panel_c']; q=fit(quad(p['u']),p['v1']); uu=p['u']; vv=p['v2'];
for name,fn in [('sat',lambda u,a: -np.expm1(-a*u*u)),('tanh2',lambda u,a:np.tanh(a*u)**2),('cos',lambda u,a:np.cos(a*u)),('rational',lambda u,a:u*u/(1+a*u*u)),('abs',lambda u,a:abs(u)**a),('tanh',lambda u,a:np.tanh(a*u))]:
 def h(u,a):return np.c_[ones(u),fn(u,a)]
 def res(a):return (vv-h(uu,a)@fit(h(uu,a),vv)).ravel()
 a=least_squares(res,.5,bounds=(.005,6)).x;print('C2',name,a,np.std(res(a)))
 if name=='sat': ac=a; cc=fit(h(uu,a),vv)
D={'raw':x,'diag':(x-c[:,0])/c[:,1],'A':z,'A_lin':(x-c[:,0])@np.linalg.inv(np.diag(c[:,1])+np.roll(np.diag(c[:,2]),1,axis=0))}
D.update({k+tag:f(v) for k,v in list(D.items()) for tag,f in [('B',lambda z:z@b[1:]+b[0]),('Bi',lambda z:(z-b[0])@np.linalg.inv(b[1:])),('C',lambda z:quad(z)@q),('BC',lambda z:quad(z@b[1:]+b[0])@q)]})
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import KFold
rates=np.array([.4153,1.8082]); ts=-np.expm1(-t[:,None]*rates); R=np.c_[Y[:,0],fit(ts,T.T).T];out=[]
for key,u in D.items():
 w=u[:,list(range(4))];fs={'Q':quad(u),'Q+sat':np.c_[quad(u),quad(np.tanh(u))[:,1:]],'Q+trig':np.c_[quad(u),np.sin(u),np.cos(u),np.sin(2*u),np.cos(2*u)],'P3':np.c_[quad(u),*[np.prod(u[:,i],axis=1) for i in it.combinations_with_replacement(range(4),3)]],'full':np.c_[quad(u),quad(np.tanh(u))[:,1:],np.sin(u),np.cos(u),np.sin(2*u),np.cos(2*u),*[np.prod(u[:,i],axis=1) for i in it.combinations_with_replacement(range(4),3)]]}
 for tag,f in fs.items():
  fitidx=np.arange(52); testidx=np.arange(52,n); F=(f-f[fitidx].mean(0))/np.maximum(f[fitidx].std(0),1e-9); L=F[:52]; V=F[52:]; M=R[:52].mean(0); Z=R[:52]-M
  for reg in [1e-6,.01,.2,2.]:
   pr=V@L.T@np.linalg.solve(L@L.T+reg*np.eye(52),Z)+M; err=[np.mean(abs(pr[:,0]-R[52:,0])),np.mean(abs(pr[:,1:]@ts.T-T[52:]))];out.append((sum(err),key,tag,reg,np.round(err,5)))
print(*sorted(out,key=lambda d:d[0])[:30],sep='\n')
