import sys,pickle,numpy as np,importlib.util
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c03_batch_10_solver as a
import dev_m102_c03_batch_00_solver as b
ss=importlib.util.spec_from_file_location('s','/holdout_task/_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(ss); ss.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.column_stack([tr['y'],tr['traj']]); n=len(x)
for name,p in [('a',s['panel_a']),('b',s['panel_b']),('c',s['panel_c'])]:
 u=p['u']; print('panel',name, 'range',u.min(),u.max())
 for key,v in p.items():
  if key=='u':continue
  for lab,f in [('linear',u),('sat',np.tanh(u)),('saturated',np.tanh(1.75*u)+.3*np.roll(np.sin(2.1*u),1,axis=1)),('cos',np.cos(2.4*u)),('exp',np.exp(-.2962*u*u))]:
   f=a._bias(f); print(key,lab, np.round(np.quantile(abs(v-f@a._ls(f,v)),[.5,.9,1]),6))
z=a._coordinates(s,x); w=a._panels(s,z); f=a._readout_features(w)
print('ranges',z.min(0),z.max(0),'readout shape',f.shape)
for ii,hi in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.arange(72)[np.arange(72)%3!=0],np.arange(72)[::3])]:
 st={**s,'train':{**tr,**{k:tr[k][ii] for k in ('x','y','traj')}},'test':{'x':x[hi]}}; ref=Y[hi]; sp=m.run_node({'__input__':st}); q=np.column_stack([sp['y_hat'],sp['state_pred']]); mult=np.r_[.5,np.full(8,.35)]
 def score(label,yp):
  d=abs(ref-yp);print(label, np.mean(d[:,0]),np.mean(d[:,1:]))
 score('fixed',q)
 for mode in ['sat','exp']:
  try:
   yp=b._panel_pred(st,1.,.3,mode);score('old '+mode,yp);score('old blended '+mode,mult*q+(1-mult)*yp)
  except Exception as e: print(e)
 pp=a.solve_batch(st); yp=np.column_stack([pp['y_hat'],pp['state_pred']]);score('archive10',yp);score('archive10 blended',mult*q+(1-mult)*yp)
 for nm,g in [('z',a._bias(z)),('f',f),('f_all_quad',a._bias(np.column_stack([w]+[w[:,i]*w[:,j] for i in range(8) for j in range(i,8)]))),('raw_cubic',a._bias(np.column_stack([z,z**2,z**3]+[z[:,i]*z[:,j] for i in range(4) for j in range(i)])))]:
  yp=g[hi]@a._ls(g[ii],Y[ii]);score(nm,yp)
