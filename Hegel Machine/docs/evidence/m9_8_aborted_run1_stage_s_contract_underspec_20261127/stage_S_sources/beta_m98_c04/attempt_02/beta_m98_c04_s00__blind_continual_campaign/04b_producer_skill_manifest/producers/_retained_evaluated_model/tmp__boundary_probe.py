from transport_probe import *
np.set_printoptions(precision=7,suppress=True)
print('endpoint data [spacing, robin, k_left, k_right, v_left, v_right]')
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); print(ax,h,s['robin_'+ax],s['kappa_'+ax][[0,-1]],s['vel_'+ax][[0,-1]])
print('telemetry column mean/std/range/correlation', np.mean(s['aux_manifold_telemetry'],0), np.std(s['aux_manifold_telemetry'],0), np.ptp(s['aux_manifold_telemetry'],0),np.corrcoef(s['aux_manifold_telemetry'].T))
N=len(idx); fun=lambda r:pred(1,'arith','local',r)-obs
fit=lambda mask,init:least_squares(lambda r:fun(r)[mask].ravel(),init,bounds=(0,np.inf),gtol=1e-12,ftol=1e-12,xtol=1e-12).x
nom=[]; series=[]; kapp=[]; adv=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); k=s['kappa_'+ax][[0,-1]]; v=s['vel_'+ax][[0,-1]]; r=s['robin_'+ax]; nom.extend([r/h]*2); series.extend(r/h/(1+r*h/(2*k))); kapp.extend(k*r/h); adv.extend(np.maximum(v*[1,-1],0)/h)
r=fit(np.ones(obs.shape,dtype=bool),nom); print('fit',r,'shift/pulse RMSE',np.sqrt(np.mean(fun(r).reshape(3,N,2)**2,axis=1)))
for name,b in [('nom/h',nom),('nom/h - inflow advection',np.array(nom)-adv),('series',series),('series - inflow advection',np.array(series)-adv),('k*nom/h',kapp)]:
 print(name,np.array(b),np.sqrt(np.mean(fun(b)**2)))
for label, groups in [('pulse',np.broadcast_to(np.arange(2),obs.shape)),('shift',np.broadcast_to(np.repeat(np.arange(3),N)[:,None],obs.shape)),('bank',np.broadcast_to(s['row_bank_id'][:,None],obs.shape)),('sensor',np.broadcast_to(np.tile(np.arange(N),3)[:,None],obs.shape))]:
 errors=[]; coefs=[]
 for g in np.unique(groups):
  test=groups==g; rb=fit(~test,nom); errors.append(np.sqrt(np.mean(fun(rb)[test]**2))); coefs.append(rb)
 print('held out',label,'worst RMSE',max(errors),'coefficient ranges',np.ptp(coefs,axis=0))
