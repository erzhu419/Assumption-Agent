import pickle,numpy as np,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(x):return np.c_[np.ones(len(x)),x]
def fit(a,y):return np.linalg.lstsq(a,y,rcond=None)[0]
def proj(a,y):return y-a@fit(a,y)
def loo(a,y): return np.sqrt(np.mean((proj(a,y)/(1-(np.linalg.qr(a)[0]**2).sum(1))[:,None])**2))
def cal(x):return np.c_[af(x),x*np.roll(x,-1,axis=1)]
a=S['panel_a'];u,v=a['u'],a['v'];w=fit(af(u),v)
for _ in range(25):
 q=np.minimum(1,.05/np.maximum(np.linalg.norm(v-af(u)@w,axis=1),1e-12))**.5;w=fit(af(u)*q[:,None],v*q[:,None])
k=np.linalg.norm(v-af(u)@w,axis=1)<2;u,v=u[k],v[k]
for i in range(4):
 x,z=u[:,i],u[:,(i+1)%4];y=v[:,i:i+1]; p=np.c_[np.ones(len(x)),x,z,x*z];print('A',i,'base',loo(p,y),fit(p,y).ravel());
 for name,q in [('squares',[x*x,z*z]),('cubes',[x**3,z**3]),('mixed',[x*x*z,x*z*z]),('all cubic',[x*x,z*z,x**3,z**3,x*x*z,x*z*z]),('sin',[np.sin(2*x),np.sin(2*z)]),('tanh',[np.tanh(2*x),np.tanh(2*z)]),('other coords',[u[:,(i+2)%4],u[:,(i+3)%4]])]:
  g=np.c_[p,np.array(q).T]; print(name,round(loo(g,y),6),fit(g,y).ravel().round(4))
 for nm,design,init in [('sat',lambda a: np.c_[p,x*z/(1+a[0]*x*x)], [.3]),('selftanh',lambda a: np.c_[p,np.tanh(a[0]*x)], [1.]),('nexttanh',lambda a: np.c_[p,np.tanh(a[0]*z)], [1.]),('pairtanh',lambda a: np.c_[p,np.tanh(a[0]*x+a[1]*z)], [1.,1.]),('pairsin',lambda a: np.c_[p,np.sin(a[0]*x+a[1]*z)], [1.,1.])]:
  opt=least_squares(lambda a:proj(design(a),y).ravel(),init,max_nfev=150); print(nm,round(loo(design(opt.x),y),6),opt.x.round(4))
C=S['panel_c'];cu=C['u'];cv=C['v1']; cf=lambda q: np.c_[af(cu),np.tanh(q*cu)]; opt=least_squares(lambda q:proj(cf(q),cv).ravel(),[2.3]*4,bounds=(.1,8));print('C per-channel tanh',opt.x,loo(cf(opt.x),cv)); print(fit(cf(opt.x),cv).round(4))
# Compare candidate coordinates using an identical smooth interpolator and the same held-out rows.
x=S['train']['x']; n=len(x); B=fit(af(S['panel_b']['u']),S['panel_b']['v']); A=fit(cal(u),v); Al=fit(af(u),v); inv=fit(af(v),u); iv0=af(x)@inv; iv=least_squares(lambda z:(cal(z.reshape(-1,4))@A-x).ravel(),iv0.ravel(),max_nfev=60).x.reshape(-1,4); print('inverse max residual, range',np.max(np.abs(cal(iv)@A-x)),iv.min(0),iv.max(0)); C1=fit(cf(opt.x),cv); C2=fit(np.c_[af(cu),np.cos(1.27*cu)],C['v2']); Y=np.c_[S['train']['y'],S['train']['traj']]; X={}
for nm,xx in [('raw',x),('A',cal(x)@A),('Ai',iv),('Ail',iv0)]:
 X[nm]=xx; X[nm+'B']=af(xx)@B; X[nm+'C1']=np.c_[af(xx),np.tanh(opt.x*xx)]@C1; X[nm+'C2']=np.c_[af(xx),np.cos(1.27*xx)]@C2
 X[nm+'BC1']=np.c_[af(X[nm+'B']),np.tanh(opt.x*X[nm+'B'])]@C1
 X[nm+'BC2']=np.c_[af(X[nm+'B']),np.cos(1.27*X[nm+'B'])]@C2
for nm,xx in X.items():
 q=(xx-xx[:52].mean(0))/xx[:52].std(0); d=cdist(q,q,'sqeuclidean'); ps=[]
 for L in [1,2,4,8]:
  r=np.sqrt(5*d)/L; K=(1+r+r*r/3)*np.exp(-r)+.1*(1+q@q.T); K.flat[::n+1]+=.0001
  p=K[52:,:52]@np.linalg.solve(K[:52,:52],Y[:52]); e=np.abs(p-Y[52:]);ps.append((round(e[:,0].mean(),4),round(e[:,1:].mean(),4)))
 print(nm,ps)
