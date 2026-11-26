import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_ep03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
ds=pickle.load(open('/holdout_task/public_alpha_ep03.pkl','rb'))[0]['dataset']
a,b,y,complete=m._prepare(ds); complete=complete[:len(y)]; lab=ds['train']['lab_id']; K=len(a.t)
print('complete per lab',[(int(c),int(np.sum(complete&(lab==c)))) for c in np.unique(lab)])
print('regime counts',[(int(np.sum(z.work>0)),int(np.sum(z.ov>0)),int(np.sum(z.ov<0))) for z in [a,b]])
for use in (False, True):
 p=np.empty_like(y[:,:K]); raw=p.copy(); rate=np.empty(len(y)); tau=[]; weight=[]
 for c in np.unique(lab):
  val=lab==c; l,t=m._fit(a,y,complete&~val,high_load=use); tau.append(np.exp(l)); weight.append(t[-2:]); mat=a.matrix(l)[...,:len(t)] @ t
  p[val]=m._predict(a,l,t,y[:,-1],complete)[val]; raw[val]=mat[val,:K]; rate[val]=mat[val,-1]
 print('model',use,'taus',np.round(tau,5),'weights',np.round(weight,5))
 for name, mask in [('lab'+str(c),lab==c) for c in np.unique(lab)] + [('subyield',(a.work==0)&(a.ov==0)),('hold+',a.ov>0),('hold-',a.ov<0),('history-only',(a.ov==0)&(a.work>0)),('missing',~complete),('complete',complete)]:
  if not np.any(mask): continue
  e=p[mask]-y[mask,:K]; r=raw[mask]-y[mask,:K]; q=rate[mask]-y[mask,-1]; rmse=lambda x: np.sqrt(np.mean(x*x))
  print(name,int(np.sum(mask)), 'RMSE corrected/raw/rate/late',np.round([rmse(e),rmse(r),rmse(q),rmse(e[:,-1])],6))
 print('overall medAE/raw medAE',np.median(abs(p-y[:,:K])),np.median(abs(raw-y[:,:K])))
print('full sample', [(k,np.asarray(v).shape,np.asarray(v).ravel()[:2]) for k,v in m.solve_alpha_ep03(ds).items()])
