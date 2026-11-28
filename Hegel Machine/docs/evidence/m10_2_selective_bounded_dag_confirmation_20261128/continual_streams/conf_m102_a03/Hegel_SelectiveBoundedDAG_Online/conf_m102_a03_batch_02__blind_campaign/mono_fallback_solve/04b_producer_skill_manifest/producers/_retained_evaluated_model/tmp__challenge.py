import pickle,runpy,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
M=runpy.run_path('/holdout_task/batch_solver.py'); fit=M['_fit']; A=M['_affine']; Q=M['_quadratic']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=(tr[k] for k in ('x','y','traj')); a,b=M['_calibrate'](s['panel_a']); z=(x-a)/b; l=A(z); q=Q(z); n=len(x); times=s['eval_times']; rng=np.random.default_rng(512)
folds=[('public',np.arange(n))]+[(f'random{k}',rng.permutation(n)) for k in range(10)]+[(label,np.argsort(v)) for label,v in [('radius',np.linalg.norm(x-x.mean(0),axis=1)),('index',np.abs(x@fit(x,y))),('state',abs(t[:,-1]))]]
for tag,p in folds:
 i,j=p[:52],p[52:]; sol=fit(l[i],t[i]); base=l[j]@sol; qr,r=np.linalg.qr(l[i]); u,sv,v=np.linalg.svd(qr.T@t[i],full_matrices=False); rank2=l[j]@np.linalg.solve(r,(u[:,:2]*sv[:2])@v[:2]); print(tag,'state',*[round(np.mean(abs(k-t[j])),6) for k in (base,rank2,.35*base+.65*rank2)],end='; ')
 orth,_=np.linalg.qr(q[i]); target=y[i]-orth@(orth.T@y[i]); d=fit(l[i],y[i])[1:]; d/=np.linalg.norm(d); starts=[np.r_[offset,c*d] for offset in (0.,-.3) for c in (.45,1.,1.8,3.,5.)]
 modes=[('rat',lambda g:g/(1+g*g)),('erf',erf),('tanh',np.tanh),('softsign',lambda g:g/np.sqrt(1+g*g)),('smoothabs',lambda g:np.sqrt(1+g*g)),('sin',np.sin)]; preds=[]
 for name,f in modes:
  def fun(h):
   v=f(l[i]@h); w=v-orth@(orth.T@v); return w*(w@target/max(w@w,1e-24))-target
  h=min((least_squares(fun,start,bounds=(-10,10),max_nfev=100) for start in starts),key=lambda s:s.cost).x
  preds.append(np.c_[q[j],f(l[j]@h)]@fit(np.c_[q[i],f(l[i]@h)],y[i]));
 blend=np.mean(np.array(preds)[:3],axis=0); old=M['_readout'](z[i],z[j],y[i]); print('y',*[round(np.mean(abs(k-y[j])),6) for k in (*preds,blend,.85*blend+.15*old,old)],flush=True)
print('telemetry',s['telemetry_vec']);print('all traj affine',fit(l,t))
