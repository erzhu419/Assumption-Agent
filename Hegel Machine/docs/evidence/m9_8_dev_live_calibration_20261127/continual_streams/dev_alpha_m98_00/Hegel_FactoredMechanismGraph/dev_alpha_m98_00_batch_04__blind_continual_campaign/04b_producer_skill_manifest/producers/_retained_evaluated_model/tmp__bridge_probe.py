import pickle,itertools,numpy as np
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
def design(x, terms):
 return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
def fit(a,y):
 w=np.ones(len(a))
 for _ in range(28):
  h=np.sqrt(w[:,None]); b=np.linalg.lstsq(a*h,y*h,rcond=None)[0]; e=y-a@b; s=1.4826*np.median(abs(e-np.median(e,axis=0)),axis=0)+1e-9; w=1/(1+np.mean((e/(2.5*s))**2,axis=1))
 return b
base=[()]+[(j,) for j in range(4)]; all2=list(itertools.combinations_with_replacement(range(4),2)); all3=list(itertools.combinations_with_replacement(range(4),3)); ax2=[(j,)*2 for j in range(4)]; ax3=[(j,)*3 for j in range(4)]
for name,terms in [('affine',base),('pure23',base+ax2+ax3),('cross2pure3',base+all2+ax3),('full3',base+all2+all3)]:
 a=design(x,terms); b=fit(a,y); e=y-a@b; norms=np.max(abs(e),axis=1); ix=np.argsort(norms); print(name,'quantiles',np.round(np.percentile(norms,[25,50,75,85,90,95,100]),6),'worst indices',ix[-9:]);
 if name!='affine':
  for t,v in zip(terms,b):
   if np.max(abs(v))>.002: print(t,np.round(v,5));
 rng=np.random.default_rng(643); ee=np.zeros(len(x));
 for te in np.array_split(rng.permutation(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),te); ee[te]=np.max(abs(y[te]-a[te]@fit(a[tr],y[tr])),axis=1)
 print('OOF quantiles',np.round(np.percentile(ee,[50,75,85,90,95]),6))
print('trial optional type',type(D.get('chamber_verification_trials')).__name__)
