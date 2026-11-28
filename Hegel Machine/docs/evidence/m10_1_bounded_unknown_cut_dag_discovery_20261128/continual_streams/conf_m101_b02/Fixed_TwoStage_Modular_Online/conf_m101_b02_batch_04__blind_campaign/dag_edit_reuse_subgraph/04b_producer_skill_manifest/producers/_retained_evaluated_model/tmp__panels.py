import numpy as np,pickle,itertools
np.set_printoptions(precision=4,suppress=True,linewidth=160)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def design(u):return np.column_stack([np.ones(len(u)),u])
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a'];D=design(a['u']); w=np.ones(len(D));
for _ in range(15):
 c=ls(D*w[:,None],a['v']*w[:,None]);r=np.linalg.norm(a['v']-D@c,axis=1);w=1/np.sqrt(1+(r/(3*np.median(r)))**2)
print('A',c,np.sort(r),sep='\n');
for k,v in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[k]['u']; y=s[k][v];x=design(u);q=np.column_stack([x,*[u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]]);
 print(k,v,'range',np.min(u,axis=0),np.max(u,axis=0),'coef affine',ls(x,y), 'RMSE affine & quad',np.mean((y-x@ls(x,y))**2,axis=0)**.5,np.mean((y-q@ls(q,y))**2,axis=0)**.5,sep='\n'); print('quad coefficients',ls(q,y));
print('traj singular',np.linalg.svd(s['train']['traj'],compute_uv=False)); print('x ranges',np.quantile(s['train']['x'],[0,.25,.5,.75,1],axis=0));print('y ranges',np.quantile(s['train']['y'],[0,.25,.5,.75,1])); print('summary',s['train']['summary'],'raw',[(x.min(),x.max(),x.mean(),x.std()) for x in (s['train']['x'],s['train']['y'],s['train']['traj'])]);
