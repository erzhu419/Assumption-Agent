exec(open('/tmp/wall_probe.py').read().split('for name,inds,lo in')[0])

def expected(axis):
 k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(len(k)-1); wrap=(k[0]+k[-1])/2
 return [0.,((k[0]+k[1])/2+h*max(-v[0],0))/k[0],(wrap+h*max(v[0],0))/k[0],0.,((k[-1]+k[-2])/2+h*max(v[-1],0))/k[-1],(wrap+h*max(-v[-1],0))/k[-1]]
for a in ('x','y'):
 print(a,expected(a))
for flags in ((0,0),(1,0),(0,1),(1,1)):
 x=np.concatenate([expected(a) if flag else x0[j*6:(j+1)*6] for j,(a,flag) in enumerate(zip('xy',flags))]); err=pred(x)-y
 print('periodic flags',flags,'RMSE by shift,pulse',np.sqrt(np.mean(err.reshape(len(shifts),-1,len(pulses))**2,axis=1)), 'max',abs(err).max())
print('public cases',[(list(e),e.get('case_id'),e['study']['reference_tracer_pulses'] is not None) for e in pickle.load(open('/holdout_task/public_batch.pkl','rb'))])
