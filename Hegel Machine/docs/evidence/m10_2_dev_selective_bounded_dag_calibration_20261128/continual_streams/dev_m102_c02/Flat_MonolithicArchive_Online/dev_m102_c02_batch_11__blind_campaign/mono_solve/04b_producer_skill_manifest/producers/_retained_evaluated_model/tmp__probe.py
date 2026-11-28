import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; x=tr['x']; t=s['eval_times']; n=52
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c02_batch_10_solver.py'); aff=m['_aff']; fit=m['_fit']; a=s['panel_a']; z=m['_calibrate'](a,x)
print('calibration error',np.sort(np.linalg.norm(m['_calibrate'](a,a['v'])-a['u'],axis=1)))
cache={'x':x,'z':z}
def score(h,Y):
 q=h[n:]@fit(h[:n],Y[:n]); return np.mean(abs(q-Y[n:])),np.sqrt(np.mean((h[:n]@fit(h[:n],Y[:n])-Y[:n])**2))
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u']; v=s[pn][vn]; ranks=[]
 models=[('linear',lambda x,p:x,[],[]),('dead',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],([0,.25],[.8,4])),('soft',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.15,0],([.1,-4],[10,4])),('power',lambda x,p:np.sign(x)*abs(x)**p[0],[1.25],([.25],[4])),('sat',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.22,2.15,.5],([0,.2,.1],[5,5,2])),('tanh',lambda x,p:np.tanh(p[0]*x+p[1]),[1,0],([.05,-4],[8,4]))]
 for nam,op,p,bd in models:
  def err(p):
   h=aff(op(u,p)); return (h@fit(h,v)-v).ravel()
  if len(p):p=least_squares(err,p,bounds=bd,gtol=1e-12).x
  e=np.sqrt(np.mean(err(p)**2)); ranks.append((e,nam,p)); cache[pn+vn+nam]=aff(op(z,p))@fit(aff(op(u,p)),v)
 print(pn,vn,sorted(ranks,key=lambda a:a[0]))
for k,W in cache.items():
 i,j=np.triu_indices(4); pair=W[:,0]*W[:,2]+W[:,1]*W[:,3]
 print(k,'linear',score(aff(W),y),'pair',score(np.c_[aff(W),pair],y),'quad',score(np.c_[aff(W),W[:,i]*W[:,j]],y),'T',score(aff(W),T))
np.savez('/tmp/cached.npz',**cache)
