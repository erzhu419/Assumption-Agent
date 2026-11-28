import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; t=tr['traj']; n=len(y); X=tr['x']; z,b,c,d,a,opp=[q[:n] for q in m._blocks(s)]; p=z*np.roll(b,1,axis=1); dd=(d*d).sum(1); cp=(c*np.roll(p,3,axis=1)).sum(1); bp=(b*p).sum(1); dx=((d*X)**2).sum(1)
base=np.c_[m._aff(b),a,opp]; saved=np.c_[base,d,dx,cp,b[:,3]*np.exp(-1.7*b[:,3]**2),bp,z[:,2]**3]
mods={'saved':saved,'rat_simple':np.c_[m._aff(b),a,opp/(1+.2*dd**2)],'rat_d':np.c_[base,d,opp/(1+.2*dd**2),dd,cp,bp],'rat_d05':np.c_[base,d,opp/(1+.5*dd**2),dd,cp,bp],'rat_d01':np.c_[base,d,opp/(1+.1*dd**2),dd,cp,bp],'rat_c':np.c_[base,c,opp/(1+.5*dd**2),cp,bp,z[:,2]**3],'rat_small':np.c_[base,opp/(1+.5*dd**2),np.sin(1.7*c[:,1]),bp,z[:,2]**3]}
g=np.c_[m._aff(b),z,a]; gs={'saved':g,'with_d':np.c_[g,d],'with_c':np.c_[g,c],'with_cd':np.c_[g,c,d],'cyclic_terms':np.c_[g,p[:,:3]],'minimal':np.c_[m._aff(b),a],'cyclic_dd':np.c_[g,d,dd]}
rng=np.random.default_rng(18403); ids=np.arange(n); chronological=[(ids[:i],ids[i:]) for i in (52,56)]; shuffled=[(np.setdiff1d(ids,j),j) for k in range(5) for j in np.array_split(rng.permutation(n),4)]; out=np.argsort(np.linalg.norm(z,axis=1)); edge=[(out[:-12],out[-12:])]; folds=chronological+shuffled+edge
for targets,models in [(y,mods),(t,gs),(t[:,-1],gs)]:
 for key,f in models.items():
  e=np.array([np.mean(abs(f[iv]@m._fit(f[it],targets[it])-targets[iv])) for it,iv in folds]); print(key,'ordered',e[:2].round(5),'shuffled mean/max',np.round([e[2:-1].mean(),e[2:-1].max()],5),'boundary',round(e[-1],5))
u,v=(s['panel_a'][k] for k in ('u','v')); r=m._aff(u); ix=np.argsort(np.linalg.norm(v-r@m._fit(r,v),axis=1))[:-3]; uu=u[ix]; vv=v[ix]; cc=np.roll(uu,-1,axis=1); ds=lambda k:np.stack([np.ones_like(uu),uu,cc,k,uu**3],axis=-1)
for label,k in [('cyclic_product',uu*cc),('neighbor_square',cc**2)]:
 f=ds(k); pred=np.column_stack([f[:,j]@m._fit(f[:,j],vv[:,j]) for j in range(4)]); print('calibration',label,'inlier RMS',np.sqrt(np.mean((pred-vv)**2)))
