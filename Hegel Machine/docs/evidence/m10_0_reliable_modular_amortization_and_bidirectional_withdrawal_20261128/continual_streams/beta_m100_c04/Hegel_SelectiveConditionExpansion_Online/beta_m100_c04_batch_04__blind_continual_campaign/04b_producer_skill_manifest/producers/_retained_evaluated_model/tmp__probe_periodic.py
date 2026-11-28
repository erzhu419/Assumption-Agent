from probe_boundary import *
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; d=i['campaign_diagnostics']; W=np.stack([x['w_pulse'] for x in d]); modes=np.einsum('ab,bij->aij',W,s['source_modes']); Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=x['y_pulse_raw']))['y_obs_raw'] for x in d]);
for ax in 'xy':
 k=s['kappa_'+ax]; v=s['vel_'+ax]; h=s['L'+ax]/(s['n'+ax]-1); print(ax,'k',k,'vel',v,'h',h,'robin',s['robin_'+ax]); print('inward face/end', (k[0]+k[1])/(2*k[0]),(k[-1]+k[-2])/(2*k[-1]), 'wrap face/end', (k[0]+k[-1])/(2*k[0]),(k[0]+k[-1])/(2*k[-1]));
def params(x):
 a,b,c,e=x
 return [a,0,c,b,0,e,1,0,0,1,0,0]
def res(x):return forward(s,params(x),modes)[0]-Y
def rms(e): return np.sqrt(np.mean(e**2,axis=0))
for hold in [-1,0,1,2]:
 tr=[j for j in range(3) if j!=hold]; z=least_squares(lambda x:res(x)[:,tr].ravel(),np.ones(4),gtol=1e-13,ftol=1e-13,xtol=1e-13); print('held',hold,'coeff',z.x,'rms',rms(res(z.x))); 
 if hold==-1:
  k=s['kappa_x']; v=s['vel_x']; h=s['Lx']/(s['nx']-1); print('physical coefficients x h²',z.x*k[[0,-1,0,-1]]); print('wrap avg + velocity',.5*(k[0]+k[-1])+h*np.maximum([v[0],-v[-1]],0)); print('inward avg + velocity',.5*(k[[0,-1]]+k[[1,-2]])+h*np.maximum([-v[0],v[-1]],0));
