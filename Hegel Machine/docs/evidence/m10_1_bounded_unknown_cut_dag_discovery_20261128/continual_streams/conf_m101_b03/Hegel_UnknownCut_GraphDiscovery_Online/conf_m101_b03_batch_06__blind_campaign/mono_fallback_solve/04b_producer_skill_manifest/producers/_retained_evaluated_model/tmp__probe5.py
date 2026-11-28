import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from graph_probe import *
u=p['u']; v=p['v2']; z=maps['ainv'];
from scipy.optimize import minimize_scalar
funcs={'cos':lambda z,k:np.cos(k*z),'sin':lambda z,k:np.sin(k*z),'tanh':lambda z,k:np.tanh(k*z),'rat':lambda z,k:z/(1+k*z*z),'rat2':lambda z,k:z*z/(1+k*z*z),'evenexp':lambda z,k:np.exp(-k*z*z),'sincos':lambda z,k:np.c_[np.cos(k*z),np.sin(k*z)]}
for name,fun in funcs.items():
 def res(k):
  f=aff(fun(u,k)); return np.sqrt(np.mean((f@np.linalg.lstsq(f,v,rcond=None)[0]-v)**2))
 grid=np.linspace(.1,12,121); i=np.argmin([res(kk) for kk in grid]); sol=minimize_scalar(res,bounds=(grid[max(i-1,0)],grid[min(i+1,120)]),method='bounded'); print('C2',name,sol.x,sol.fun)
 if sol.fun<.025:
  f=aff(fun(u,sol.x)); d=np.linalg.lstsq(f,v,rcond=None)[0]; print('C2 coefficients',d); maps['ainvC2_'+name]=aff(fun(z,sol.x))@d
for k in ['raw','ainv','ainvb']:
 z=maps[k]; c2=maps['ainvc']; F=[aff(z),z*z,np.array([z[:,i]*z[:,j] for i,j in pairs]).T,z**3]; print('target correlations',k,np.round(np.corrcoef(np.c_[*F[1:], z, Y].T)[:-3,-3:],2));
 for j in range(4):
  for l in range(j+1,4):
   for sg in [-1,1]:
    w=z[:,j]+sg*z[:,l]; F.extend([np.sin(2.4*w)[:,None],np.cos(2.4*w)[:,None],(w/(1+w*w))[:,None]])
 for scale in [.5,1.,2.,3.,5.]:
  F.extend([np.sin(scale*z),np.cos(scale*z),np.tanh(scale*z),z/(1+scale*z*z),(z*z)/(1+scale*z*z)])
 ff=np.column_stack(F); yy=Y[:n]-Y[:n].mean(0); ff=ff[:,1:]; mu=ff[:n].mean(0); sd=ff[:n].std(0); ff=(ff-mu)/sd
 for rr in [.001,.01,.1,1.]:
  f=ff[:n]; pred=ff[n:]@np.linalg.solve(f.T@f+rr*np.eye(f.shape[1]),f.T@yy)+Y[:n].mean(0); print(k,rr,np.mean(abs(pred-Y[n:]),0),np.mean(abs(pred[:,1:]@basis(r).T-T[n:])))
