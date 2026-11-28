exec(open('/tmp/stress.py').read().split('for label,j in sets:')[0])
base=fs[:,:9]; xx=fs[:,9:].reshape(len(z),4,4); diag=xx[:,np.arange(4),np.arange(4)]; dot=diag.sum(1)
designs={'full':fs,'diag':np.c_[base,diag],'diag+10':np.c_[base,diag,xx[:,1,0]],'dot+10':np.c_[base,dot,xx[:,1,0]]}
for reg in (.0003,.001,.003,.01):
 designs[f'full ridge {reg}']=fs
for name,d in designs.items():
 metrics=[]
 for label,j in sets:
  i=np.setdiff1d(np.arange(len(z)),j); trj=tr['traj'][i]; u,sw,vt=np.linalg.svd(trj,full_matrices=False); temporal=vt[:2]; h=trj@temporal.T
  a=d[i]; b=d[j]
  if 'ridge' in name:
   lam=float(name.split()[-1]); S=np.diag(np.r_[np.zeros(9),np.full(16,np.sqrt(lam))]); a=np.vstack([a,S]); h=np.vstack([h,np.zeros((len(S),2))]);
  p=(b@m._ls(a,h))@temporal; e=abs(p-tr['traj'][j]); metrics.append([e.mean(),e.mean(0).max(),e.max()])
 print(name,'required',np.round(metrics[0],5),'max',np.round(np.max(metrics,axis=0),5),'avg',np.round(np.mean(metrics,axis=0),5))
