exec(open('/tmp/probe_wall.py').read().split("print('Robin');")[0])
for key in 'xy':
 k,v=st['kappa_'+key],st['vel_'+key]; h=st['L'+key]/(len(k)-1)
 wrap=(k[0]+k[-1])/(2*h*h)
 print(key,'endpoint velocities',v[[0,-1]],'face wrap',wrap)
 print('adjacent', (k[0]+k[1])/(2*h*h)+max(-v[0],0)/h,(k[-2]+k[-1])/(2*h*h)+max(v[-1],0)/h,'wrap upwind',wrap+max(v[0],0)/h,wrap+max(-v[-1],0)/h)
p=[]
for key in 'xy':
 k,v=st['kappa_'+key],st['vel_'+key]; h=st['L'+key]/(len(k)-1)
 wrap=(k[0]+k[-1])/(2*h*h)
 p.extend([0,0,((k[0]+k[1])/2+h*max(-v[0],0))/k[0],((k[-2]+k[-1])/2+h*max(v[-1],0))/k[-1],wrap+max(v[0],0)/h,wrap+max(-v[-1],0)/h])
print('Periodic closure coefficients',p); report(p)
Y=np.array([calibrate_upstream({**st,'y_obs_raw':pulse['y_pulse_raw']})['y_obs_raw'] for pulse in item['campaign_diagnostics']]); print('Dynamic rosette fit'); report(p)
e=predict(p)-Y; print('pulse x shift RMSE',np.sqrt(np.mean(e.reshape(len(W),len(st['mu_shifts']),-1)**2,axis=2)))
