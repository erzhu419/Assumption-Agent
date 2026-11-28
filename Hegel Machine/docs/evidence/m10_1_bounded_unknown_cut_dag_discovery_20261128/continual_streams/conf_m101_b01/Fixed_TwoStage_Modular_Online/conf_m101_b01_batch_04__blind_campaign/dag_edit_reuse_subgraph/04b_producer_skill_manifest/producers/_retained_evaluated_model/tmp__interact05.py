from separable05 import *
from scipy.optimize import least_squares

def mae(A,Y):
 b=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0];return round(np.mean(abs(Y[52:]-A[52:]@b)),5)
def feats(x,k,mode):
 a=np.arctan(k*x); q=x/(1+.16*x*x); e=x*x/(1+.16*x*x)
 aa=poly(a,2); ax=poly(x,2)
 return {'Qatan':aa,'Qx+atan':np.column_stack((ax,a)),'Qatan+x':np.column_stack((aa,x)),'Qatan+e':np.column_stack((aa,e)),'Qatan+e+x':np.column_stack((aa,e,x)),'Qatan+Qx':np.column_stack((aa,ax[:,1:])),'Catan':poly(a,3),'Catan+x+e':np.column_stack((poly(a,3),x,e)),'Qboost':np.column_stack((ax,a,q,e)), 'hybrid':np.column_stack((ax[:,:5],a,e,[(x[:,i]*a[:,j]) for i in range(4) for j in range(i+1,4)] if False else np.stack([a[:,i]*x[:,j]+a[:,j]*x[:,i] for i in range(4) for j in range(i+1,4)],1)))}[mode]
for mode in ('Qatan','Qx+atan','Qatan+x','Qatan+e','Qatan+e+x','Qatan+Qx','Catan','Catan+x+e','Qboost','hybrid'):
 for k in [1.1,2.1,3.2]:
  A=feats(z,k,mode);print(mode,k,score(A,y),mae(A,y))
 def resid(k):
  A=feats(z[:52],k,mode);return y[:52]-A@np.linalg.lstsq(A,y[:52],rcond=None)[0]
 o=least_squares(resid,np.repeat(2.1,4),bounds=(.3,5),max_nfev=80);A=feats(z,o.x,mode);print('opt',mode,np.round(o.x,3),score(A,y),mae(A,y))

def vp_panel(x,Y,name,fn,starts,bounds):
 def r(k):
  A=np.column_stack((np.ones(len(x)),fn(x,k)));return (Y-A@np.linalg.lstsq(A,Y,rcond=None)[0]).ravel()
 o=least_squares(r,starts,bounds=bounds,max_nfev=400); print(name,np.round(o.x,5),np.sqrt(np.mean(r(o.x)**2)));return o.x
pc=raw['panel_c']; u=pc['u'];
for name,fn,start,lo,hi in [
 ('shift atan',lambda x,k:np.arctan(k[0]*(x+k[1])),[2.1,0],[.1,-1],[8,1]),
 ('atan plus linear',lambda x,k:np.arctan(k[0]*x)+k[1]*x,[2.1,0],[.1,-1],[8,1]),
 ('shift power',lambda x,k:(x+k[2])/(1+k[0]*(x+k[2])**2)**k[1],[1,.5,0],[.01,.05,-1],[30,5,1]),
 ('sinh saturate',lambda x,k:np.arcsinh(k[0]*x)+k[1]*x,[3,0],[.1,-2],[20,2]),
 ('shift 4 atan',lambda x,k:np.arctan(k[:4]*(x+k[4:])),[2.1]*4+[0]*4,[.1]*4+[-1]*4,[8]*4+[1]*4),
 ('odd fraction',lambda x,k:(x+k[0]*x**3)/(1+k[1]*x*x+k[2]*x**4),[.1,1,.1],[-5,0,0],[10,30,30]),
 ('tanh plus sin',lambda x,k:np.tanh(k[0]*x)+k[1]*np.sin(k[2]*x),[2,.2,1],[.1,-2,.1],[8,2,8])]: vp_panel(u,pc['v1'],name,fn,start,(lo,hi))
for name,fn,start,lo,hi in [
 ('v2 numerator quadratic',lambda x,k:(x+k[0]*x*x)/(1+k[1]*x*x),[.45,.02],[-2,-.1],[2,3]),
 ('v2 affine plus rational',lambda x,k:x+k[0]*x*x/(1+k[1]*x*x),[.45,.16],[-2,0],[2,3]),
 ('v2 affine plus cos',lambda x,k:x+k[0]*(1-np.cos(k[1]*x)),[.7,1.2],[-5,.1],[10,8])]:vp_panel(u,pc['v2'],name,fn,start,(lo,hi))
for k in [2,2.1,2.2,2.4]:
 A=F(z,'atan',np.repeat(k,4),True); print('traj fixed',k,score(A,T),mae(A,T))
