import pickle, numpy as np, runpy, glob
from scipy.optimize import minimize_scalar
F=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); A=runpy.run_path(sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py'))[-1]); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=tr['traj']; a=A['affine']; fit=A['fit']; gp=A['gp']; cx,v0=F['_coordinates'](s,x).T.reshape(2,4,len(x)).transpose(0,2,1); b,c=s['panel_b'],s['panel_c']; bx=a(x)@fit(a(b['u']),b['v']); u=c['u'];
def cal(v,form,bounds):
 def obj(k):
  X=form(u,k); return np.linalg.norm(X@fit(X,v)-v)
 k=minimize_scalar(obj,bounds=bounds,method='bounded',options={'xatol':1.e-12}).x; print('panel cal',k,obj(k)); return k,fit(form(u,k),v)
vform=lambda q,k:a(np.arctan(k*q)); wform=lambda q,k:np.c_[a(q),q*q/(1+k*q*q)]; k,V=cal(c['v1'],vform,(.1,6)); h,W=cal(c['v2'],wform,(0,3)); w=lambda q:wform(q,h)@W; v=lambda q:vform(q,k)@V
Q={'raw':x,'A':cx,'B':bx,'Av':np.c_[cx,v(cx)],'Aatan':np.c_[cx,vform(cx,k)[:,1:]],'Bw':np.c_[bx,w(bx)],'Bsq':wform(bx,h)[:,1:],'Bwonly':w(bx),'Xw':np.c_[x,w(x)],'Aw':np.c_[cx,w(cx)],'Bv':np.c_[bx,v(bx)],'BAv':np.c_[bx,v(cx)],'Bwv':np.c_[bx,w(bx),v(cx)]}
for name,q in Q.items():
 for add in (False,True):
  results=[]; scores=[]
  for va in (np.arange(52,72),np.arange(0,20),np.arange(26,46),np.random.default_rng(726).permutation(len(x))[:20]):
   ti=np.setdiff1d(np.arange(len(x)),va); _,_,vt=np.linalg.svd(z[ti],full_matrices=False); coeff=z[ti]@vt[:2].T; pred=np.column_stack([gp(q[ti],q[va],y,add) for y in coeff.T]); res=[np.abs(pred[:,:r]@vt[:r]-z[va]).mean() for r in (1,2)]; results.append(res); scores.append(np.mean(np.abs(pred-z[va]@vt[:2].T),0))
  print(name,add,np.round(results,4).tolist(),np.round(scores,4).tolist(),flush=True)
