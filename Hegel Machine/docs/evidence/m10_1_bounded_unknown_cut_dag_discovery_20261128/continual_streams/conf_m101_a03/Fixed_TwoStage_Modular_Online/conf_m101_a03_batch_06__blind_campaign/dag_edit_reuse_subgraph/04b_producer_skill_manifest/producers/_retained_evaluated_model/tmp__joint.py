import sys,pickle,numpy as np,itertools,runpy
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task');import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q=n._inverse_a(tr['x'],n._fit_panel_a(s['panel_a'])); b=n._fit_panel_b(s['panel_b']); t=tr['traj']; y=tr['y']; pairs=list(itertools.combinations_with_replacement(range(4),2)); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; aff=lambda z:np.column_stack((np.ones(len(z)),z)); quad=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in pairs]); basic=lambda z:np.column_stack((aff(z),z[:,0]*z[:,2],z[:,1]*z[:,3])); base=np.array([0,1,1,1,-1,0,0]); rng=np.random.default_rng(130)
for split in (np.arange(52),):
 te=np.setdiff1d(np.arange(len(q)),split); Q=quad(q); c=np.linalg.lstsq(np.column_stack((aff(q),Q))[split],t[split],rcond=None)[0]; U,S,V=np.linalg.svd(c[5:],full_matrices=False); k=U[:,0]/U[1,0]; g=Q@k
 Qs=np.column_stack([Q[:,1]+Q[:,8],quad(q@b[1:])[:,3]+quad(q@b[1:])[:,5],quad(q@b[1:])[:,2]+quad(q@b[1:])[:,6],quad(q@b[1:])[:,1]+quad(q@b[1:])[:,8]])
 print('carrier coefficients for simple combinations:',np.linalg.lstsq(Qs,k@Q.T,rcond=None)[0],flush=True)
 for name,z in [('q',q),('Bf',q@b[1:]+b[0]),('Bi',(q-b[0])@np.linalg.inv(b[1:]))]:
  zz=z[split]; gg=g[split]; yy=y[split]; tt=t[split]; C=np.linalg.lstsq(np.column_stack((aff(zz),gg)),tt,rcond=None)[0]; L=C[1:5]; dirs=[np.linalg.solve(L@L.T+r*r*np.eye(4),L@C[5]) for r in [.03,.1,.3]]; dirs += [np.eye(4)[j]*np.dot(L[j],C[5])/np.dot(L[j],L[j]) for j in range(4)]
  results=[]
  def fit(p,idx):
   a=basic(z[idx]+g[idx,None]*p[:4]); return np.column_stack((a,np.tanh(a@p[4:])))
  def res(p):
   a=fit(p,split); w=np.linalg.lstsq(a,yy,rcond=None)[0]; A=a[:,:5]; return np.r_[a@w-yy, (A@np.linalg.lstsq(A,tt,rcond=None)[0]-tt).ravel(), .004*p]
  for j,d in enumerate(dirs):
   for i in range(5):
    v=base if i==0 else rng.normal(0,.8,7)
    sol=least_squares(res,np.r_[d,v],max_nfev=180); p=sol.x; a=fit(p,split); e=np.mean(abs(fit(p,te)@np.linalg.lstsq(a,yy,rcond=None)[0]-y[te])); results.append((np.linalg.norm(sol.fun),e,p))
  for train,e,p in sorted(results,key=lambda u:u[0])[:8]:
   Z=z+g[:,None]*p[:4]; o=sink({'node_00':{'z':Z[split],'z_test':Z[te],'y':yy,'traj':tt}}); print(name,'train',round(train,5),'profile CV',round(e,5),'sink CV',round(np.mean(abs(o['y_hat']-y[te])),5),round(np.mean(abs(o['state_pred']-t[te])),5),'d and dir',np.round(p,4).tolist(),flush=True)
