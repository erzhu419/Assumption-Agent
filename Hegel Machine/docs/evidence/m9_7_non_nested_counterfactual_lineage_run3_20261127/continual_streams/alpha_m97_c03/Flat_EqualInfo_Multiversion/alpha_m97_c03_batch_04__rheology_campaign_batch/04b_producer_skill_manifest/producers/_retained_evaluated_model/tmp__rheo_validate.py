import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
for mode in ('active','all_log'):
 pp,w=m._fit(args(slice(None)),y,mode); e=m._basis(pp,*args(slice(None)),terminal_mode=mode)@w-y
 print(mode,'train',np.exp(pp),w,'metrics',metrics(e))
 for name,g in [('lab',lab),('stratum',st)]:
  for k in np.unique(g):
   v=g==k; pp,w=m._fit(args(~v),y[~v],mode); e=m._basis(pp,*args(v),terminal_mode=mode)@w-y[v]; print(name,k,'taus',np.exp(pp),'metrics',metrics(e),'active',metrics(e[D[v]>0])[:2],'inactive',metrics(e[D[v]==0])[:2]);
