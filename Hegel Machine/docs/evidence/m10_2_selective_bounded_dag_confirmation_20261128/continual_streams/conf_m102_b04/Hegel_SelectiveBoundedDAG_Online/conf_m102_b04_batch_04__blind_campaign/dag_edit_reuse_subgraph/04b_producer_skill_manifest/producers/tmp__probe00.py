import pickle,numpy as np,runpy
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
arc=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b04_batch_01_solver.py');fit=arc['fit']; aff=arc['affine']
fs={'atan':lambda x,a:np.arctan(a*x),'tanh':lambda x,a:np.tanh(a*x),'sin':lambda x,a:np.sin(a*x),'sqrt':lambda x,a:x/np.sqrt(1+a*x*x),'rat':lambda x,a:x/(1+a*x*x),'quad':lambda x,a:x*x/(1+a*x*x),'cos':lambda x,a:np.cos(a*x),'exp':lambda x,a:np.exp(-a*x*x)}
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x=s[p]['u'];y=s[p][k]; print(p,k)
 for name,f in fs.items():
  for base in [False,True]:
   def loss(a,full=False):
    d=aff(np.c_[x,f(x,a)] if base else f(x,a));r=y-d@fit(d,y);r=r/(1-(np.linalg.svd(d,full_matrices=False)[0]**2).sum(1))[:,None];return np.mean(abs(r))
   g=np.linspace(.03,5,81);i=np.argmin([loss(a) for a in g]);a=minimize_scalar(loss,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded').x; print(name,base,round(a,5),round(loss(a),6))
tr=s['train']; z=tr['traj']; v=np.linalg.svd(z,full_matrices=False)[2][:2]; Y=np.c_[tr['y'],z@v.T]; n=52
cx,bx,w,q=arc['coordinates'](s,tr['x']);xx={'x':tr['x'],'cx':cx,'bx':bx,'v1':w,'v2':q,'cxv1':np.c_[cx,w],'xv2':np.c_[tr['x'],q],'bxv2':np.c_[bx,q],'everything':np.c_[cx,w,q]}
for name,x in xx.items():
 for add in [False,True]:
  m=np.column_stack([arc['gp'](x[:n],x[n:],y[:n],add) for y in Y.T]);print('GP',name,add,np.mean(abs(m[:,0]-Y[n:,0])),np.mean(abs(m[:,1:]@v-z[n:])),np.mean(abs(m-Y[n:]),axis=0))
