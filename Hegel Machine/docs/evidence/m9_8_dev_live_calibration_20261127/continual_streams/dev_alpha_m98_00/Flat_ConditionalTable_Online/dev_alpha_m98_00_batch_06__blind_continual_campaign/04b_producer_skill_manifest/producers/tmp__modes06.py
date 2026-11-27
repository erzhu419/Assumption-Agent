import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber06 import *
from scipy.stats import pearsonr
n=len(h); delta=np.c_[y-base@F.T,r-base@(1/tau)]
K=np.c_[F,t].T; K=np.column_stack((K,np.r_[1/tau,1])); w=np.r_[np.ones(len(t)),2.]; amps=np.linalg.lstsq((K*w).T,(delta*w).T,rcond=None)[0].T
baseline=dr[:,None]*np.r_[tau*np.array([.52,.43]),.05]; extra=amps-baseline
print('delta branch and rate signature'); print(np.round(np.c_[dr,extra,extra@K[:,-1],extra[:,0]/tau[0]+extra[:,1]/tau[1],p[:,-2:],h],4))
features={'constant':np.ones(n),'hold':h,'drift':dr,'drift abs':abs(dr),'drift sq':dr**2,'mean':p.mean(1),'last-first':p[:,-1]-p[:,0], 'last-prev':p[:,-1]-p[:,-2], 'pstd':p.std(1),'dr hold':dr*h,'dr mean':dr*p.mean(1)}
for k in range(p.shape[1]):
 features[f'p{k}']=p[:,k];features[f'p{k} sq']=p[:,k]**2;features[f'dr p{k}']=dr*p[:,k];features[f'hold-p{k}']=h-p[:,k]
for k in range(2): features[f'z{k}']=Z[:,k]; features[f'gap{k}']=h-Z[:,k]; features[f'gap{k} sq']=(h-Z[:,k])**2
for k in range(1,5):features[f'x{k}']=x[:,k];features[f'x{k} sq']=x[:,k]**2
for q in ('hold_channel_raw','past_channel_raw'):
 L=invert(g[q],co)
 if L.ndim==3:L=L[:,-1]
 for k in range(4):features[f'{q[:4]} ch{k}']=L[:,k]
for name,val in [('fast',extra[:,0]),('slow',extra[:,1]),('slope',extra[:,2]),('rate',extra@K[:,-1])]:
 v=sorted([(abs(c),c,k) for k,j in features.items() if np.std(j)>1e-8 for c in [pearsonr(val,j).statistic]],reverse=True);print(name,[(k,round(c,3)) for _,c,k in v[:9]])
# Evaluate plausible segment-resolved history, not just equilibrated branch state.
b0=sum(mk(dr*tau[j]*a,j) for j,a in enumerate([.52,.43]))+.05*mk(dr,2)
for names in [['p0','p1','p2','p3','p4','p5'],['p4','p5'],['last-prev'],['last-first'],['z0','z1'],['hold','mean'],['p0','p5']]:
 evaluate('history '+str(names),[mk(features[k],j) for k in names for j in range(2)],b0)
for k in ('drift abs','drift sq','pstd','hold','hold-p5','gap1 sq','dr hold','dr mean'):
 evaluate(k,[mk(features[k],j) for j in range(2)],b0)
