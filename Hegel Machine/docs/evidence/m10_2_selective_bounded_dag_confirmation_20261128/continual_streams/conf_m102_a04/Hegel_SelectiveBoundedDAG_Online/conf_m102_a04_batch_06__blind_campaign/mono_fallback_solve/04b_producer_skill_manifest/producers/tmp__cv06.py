import sys, pickle, importlib.util, numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y, t=tr['y'],tr['traj']; z=m._coords(s['panel_a'],tr['x']); c,d=m._panels(s['panel_c'],z); F=m._library(z,c,d)
rng=np.random.default_rng(51); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(8)]+[np.argsort(k) for k in (abs(z).max(1), np.linalg.norm(z,axis=1))]
def evaluate(fn):
    out=[]
    for p in splits:
        a,b=p[:52],p[52:]; e, h=fn(a,b); out.append((abs(e-y[b]).mean(), abs(h-t[b]).mean(), abs(h-t[b]).mean(0).max()))
    return np.array(out)
if __name__=='__main__':
 for r in (2,3):
  for k in (32,36,39,43):
   E=evaluate(lambda a,b:m._sparse(F[np.r_[a,b]],y[a],t[a],r,k)); print(r,k,'public',E[0].round(5),'random mean/max',E[1:9].mean(0).round(5),E[1:9].max(0).round(5),'extrap max',E[9:].max(0).round(5))
