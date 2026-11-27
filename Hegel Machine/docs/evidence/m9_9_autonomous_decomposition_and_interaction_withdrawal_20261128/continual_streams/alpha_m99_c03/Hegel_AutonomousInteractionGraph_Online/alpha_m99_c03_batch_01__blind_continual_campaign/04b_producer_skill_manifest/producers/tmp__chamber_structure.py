import numpy as np,pickle
from scipy.linalg import lstsq
z=np.load('/tmp/chamber_probe.npz'); p,h,x,dt,t,tau,th,c=[z[k] for k in ('past','hold','features','dt','t','taus','theta','corr')]; n=len(h)
np.set_printoptions(precision=5,suppress=True,linewidth=140)
after=np.cumsum(dt[::-1])[::-1]-dt
mem=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-after[:,None]/tau)); g=h[:,None]-mem
b=z['bridge']; it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; q=it['campaign_diagnostics']; raw=q['past_channel_raw']; ph=raw@b[1:]+b[0]; hh=q['hold_channel_raw']@b[1:]+b[0]
for name,block in [('train',d['train']),('test',d['test']),('diag',q)]:
 print(name,{k:(np.shape(v),type(v).__name__) for k,v in block.items()}); print('labels',np.unique(block['lab_id'],return_counts=True),np.unique(block['stratum'],return_counts=True));
 if 'aux_chamber_telemetry' in block:
  a=block['aux_chamber_telemetry']; print('aux',str(a)[:650]); print('mean',np.mean(a,axis=0))
F=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]])
print('latent svd',np.linalg.svd(c)[1:]); print('latent covariance vectors',np.linalg.svd(c-c.mean(0))[1:]); print('latent drift vs fast/slow',lstsq(np.c_[np.ones(n),c[:,:2]],c[:,2])[0]);
features={};
def add(name,a): features[name]=np.asarray(a)
add('hold',h); add('prev',p[:,-1]); add('jump',h-p[:,-1]); add('histmean',p@dt/dt.sum()); add('histstd',p.std(1)); add('histpower',p**2@dt/dt.sum()); add('lastdelta',p[:,-1]-p[:,-2]); add('TV',abs(np.diff(p,axis=1)).sum(1)); add('absjump',abs(h-p[:,-1]));
for i in range(2): add('m'+str(i),mem[:,i]); add('g'+str(i),g[:,i]); add('sqmemory'+str(i),(p**2)@(-np.expm1(-dt/tau[i])*np.exp(-after/tau[i]))); add('gap2_memory'+str(i),((h[:,None]-p)**2)@(-np.expm1(-dt/tau[i])*np.exp(-after/tau[i])));
for i in range(p.shape[1]): add('p'+str(i),p[:,i]); add('p_sq'+str(i),p[:,i]**2)
for i in range(4):
 add('x'+str(i),x[:,i]); add('x_sq'+str(i),x[:,i]**2); add('holdch'+str(i),hh[:,i]); add('prevch'+str(i),ph[:,-1,i]); add('jumpch'+str(i),hh[:,i]-ph[:,-1,i]); add('lab'+str(i),q['lab_id']==i)
for name,a in list(features.items()):
 if name.startswith(('x_sq','lab')):continue
 for j in range(4): add(name+'*x'+str(j),a*x[:,j])
 for v in ('g0','g1','hold','jump','prev'):
  add(name+'*'+v,a*features[v]);
for name,Y in [('fast',c[:,0]),('slow',c[:,1]),('drift',c[:,2]),('observed initial residual',z['er']),('observed end residual',z['ec'][:,-1])]:
 print('\n',name); A=np.ones((n,1)); r=Y-Y.mean(); seen=[]
 for k in range(7):
  ranks=[]
  for s,f in features.items():
   u=f-A@lstsq(A,f)[0]; norm=np.linalg.norm(u)
   if norm>1e-8: ranks.append((abs(u@r)/norm,s))
  ranks.sort(reverse=True); _,s=ranks[0]; A=np.c_[A,features[s]]; seen.append(s); coef=lstsq(A,Y)[0]; r=Y-A@coef
  print(k,seen[-1], 'rmse',round(np.std(r),5),'top',ranks[:3], 'coef',coef if k in (2,5) else '')
