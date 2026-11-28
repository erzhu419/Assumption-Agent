import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
def fit(d,y):
 w=np.linalg.lstsq(d,y,rcond=None)[0]; r=d@w-y; return (np.mean(r*r)**.5,w)
u,v=(s['panel_a'][k] for k in ['u','v']);m=np.max(abs(v),axis=1)<3;u,v=u[m],v[m];p=np.roll(u,1,axis=1);n=np.roll(u,-1,axis=1)
for typ,g in [('sin',np.sin),('tanh',np.tanh)]:
 def est(j,k):return fit(np.column_stack([np.ones(len(u)),u[:,j],p[:,j],u[:,j]*n[:,j],g(k*u[:,j])]),v[:,j]);
 for j in range(4):
  res=minimize_scalar(lambda a:est(j,a)[0],bounds=(.07,2),method='bounded');print('A',typ,j,res.x,est(j,res.x), 'at .22',est(j,.22))
for panel,out in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=(s[panel][k] for k in ['u',out]);a=np.column_stack([np.ones(len(u)),u]); print(panel,out)
 for typ,g,lo,hi in [('sin',np.sin,.1,5),('tanh',np.tanh,.1,6),('asinh',np.arcsinh,.1,8),('sin half',lambda x:np.sin(x)+0.5*np.sin(2*x),.1,5),('tanh cube',lambda x:np.tanh(x**3),.1,4),('square',np.square,.8,1.2),('softclip',lambda x:np.clip(x,-1,1),.8,8)]:
  def est(k):return fit(np.column_stack([a,g(k*u)]),v)
  grid=np.linspace(lo,hi,121);i=np.argmin([est(k)[0] for k in grid]);o=minimize_scalar(lambda k:est(k)[0],bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded');print(typ,round(o.x,5),round(o.fun,7));
  if (panel,typ) in [('panel_b','sin'),('panel_b','tanh'),('panel_c','square'),('panel_c','asinh')]:print(est(o.x)[1])
