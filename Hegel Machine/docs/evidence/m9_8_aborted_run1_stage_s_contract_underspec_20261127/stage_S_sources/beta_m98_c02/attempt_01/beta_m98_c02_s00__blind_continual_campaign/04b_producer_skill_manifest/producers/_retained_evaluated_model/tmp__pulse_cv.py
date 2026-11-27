from transport_probe import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];print('public item keys',list(s));s=s['study'];cal=calibrate(s)
ps=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in ps],axis=1);Y=np.stack([cal(p['y_pulse_raw']) for p in ps],axis=1)
for adv in ['row','rowface']:
 for face in ['arith','harm']:
  predict,B,h=prepare(s,adv,face,-1); nom=np.repeat([s['robin_x'],s['robin_y']],2)/np.repeat(h,2)
  n=len(s['sensor_ij']); group=np.tile(np.arange(n)//4,len(s['mu_shifts']));shift=np.repeat(np.arange(len(s['mu_shifts'])),n)
  for mode in ['nominal','axis','wall']:
   p0={'nominal':np.array([]),'axis':nom[::2],'wall':nom}[mode]
   expand=lambda p:nom if mode=='nominal' else np.repeat(p,2) if mode=='axis' else p
   def score(tr,te):
    def resid(p,m):return (predict(expand(p),B@W)[0]-Y)[m].ravel()
    p=least_squares(resid,p0,args=(tr,),bounds=(0,np.inf),gtol=1e-12).x if len(p0) else p0
    return np.sqrt(np.mean(resid(p,te)**2)),expand(p)
   allm=np.ones_like(Y,dtype=bool); v=[score(np.broadcast_to(group[:,None]!=i,Y.shape),np.broadcast_to(group[:,None]==i,Y.shape))[0] for i in np.unique(group)];u=[score(np.broadcast_to(shift[:,None]!=i,Y.shape),np.broadcast_to(shift[:,None]==i,Y.shape))[0] for i in np.unique(shift)];t=[score(allm & (np.arange(Y.shape[1])!=i),allm & (np.arange(Y.shape[1])==i))[0] for i in range(Y.shape[1])];print(adv,face,mode,'all',score(allm,allm),'worst group/shift/pulse',np.round([max(v),max(u),max(t)],8),'individual pulse',np.round(t,8))
print('archives',__import__('os').path.isdir('/tmp/hegel_continual_archive'))
