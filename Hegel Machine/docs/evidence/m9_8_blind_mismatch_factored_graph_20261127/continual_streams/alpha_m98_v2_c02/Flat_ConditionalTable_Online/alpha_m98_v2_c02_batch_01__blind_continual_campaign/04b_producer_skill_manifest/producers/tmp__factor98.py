exec(open('/tmp/law98.py').read().split('for mode in range')[0])
lab=a['lab_id']; ids=np.arange(len(h)); sw=d['calibration_load_cells']['single_channel_sweeps']; slopes=[]; offsets=[]
for g in np.unique(lab):
 k=sw['lab_id']==g; slope,offset=np.polyfit(sw['true_stress'][k],sw['cmd_stress'][k],1); slopes.append(slope); offsets.append(offset)
gain=np.array(slopes)[lab]; offset=np.array(offsets)[lab]; print('sweep gain offset',slopes,offsets)

def fit2(idx,Xs,Xf,free=True,drift=False):
 def calc(p,ret=False):
  H=[expb(p[0]),expb(p[1]),expb(p[2])] if free else [*T,expb(p[0])]; X=[Xs,Xs,Xf];
  if drift:X.append(Xf);H.append(basis(u)[2]);
  A=np.concatenate([x[:,None,:]*v[None,:,None] for x,v in zip(X,H)],axis=2); co=lstsq(A[idx].reshape(-1,A.shape[2]),Y[idx].ravel())[0]; E=A@co-Y
  return (E,co) if ret else E[idx].ravel()
 z=least_squares(calc,[u[0],u[1],2.3] if free else [2.3],bounds=([.35,1.5,.4],[.85,24,12]) if free else ([.4],[12]),max_nfev=100).x
 return (*calc(z,True),z)
for label,Xs,Xf,free,drift in [('full',np.c_[h,s],f,True,False),('fixed',np.c_[h,s],f,False,False),('linear',np.c_[h,s],f,False,True),('sweep forward',np.c_[h,s]*gain[:,None]+offset[:,None],f,False,False),('sweep inverse',(np.c_[h,s]-offset[:,None])/gain[:,None],f,False,False),('feature labs',np.c_[h,s],np.c_[f,gain*q,offset*q],False,False)]:
 E,C,z=fit2(ids,Xs,Xf,free,drift); loo=np.stack([fit2(ids[ids!=i],Xs,Xf,free,drift)[0][i] for i in ids]); print(label,'tau',z,'C',C,'fit',rm(E),'LOO',rm(loo),'worst lab',max(np.sqrt(np.mean(loo[lab==k]**2)) for k in np.unique(lab)))
for g in np.unique(lab):print('group',g,'n',sum(lab==g),'feat missing',np.isnan(a['coupon_features_raw'][lab==g]).sum(0))
