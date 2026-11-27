from probe_transport import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; edge,idx=setup(s); A=build(s,1,0,1); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); B=s['source_modes'].reshape(len(W),-1,order='F').T@W; Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); base=np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2); print('nominal robin / h',base); print('nominal RMSE',np.sqrt(np.mean((pred(A,edge,idx,mu,B,base)-Y)**2,axis=0))); print('solved row layout by shift',len(idx),Y.shape); t=s['sensor_calibration_sweeps']['ref_rosette']; print('reference sweeps groups',[(int(i),int(np.sum(np.count_nonzero(t,axis=1)==i))) for i in range(5)]); print('rosettes 16:36',t[16:36]); print('rosette ranges',t.min(0),t.max(0)); print('telemetry mean/cov',np.mean(s['aux_manifold_telemetry'],axis=0),np.cov(s['aux_manifold_telemetry'].T))
xx,yy=np.indices((nx,ny)); x=(xx.ravel(order='F')/(nx-1)-.5)*2; y=(yy.ravel(order='F')/(ny-1)-.5)*2; Z=s['Z_nuisance']; bank=np.eye(4)[s['row_bank_id']]; rms=lambda a:np.sqrt(np.mean(a*a)); E2=np.column_stack((edge[:,:2].sum(1),edge[:,2:].sum(1))); corner=(E2[:,0]*E2[:,1])[:,None]
for tag,E,N in [('symmetric',E2,None),('walls',edge,None),('walls+slopes',np.column_stack((edge,edge[:,:2]*y[:,None],edge[:,2:]*x[:,None])),None),('walls+corner',np.column_stack((edge,corner)),None),('walls+rosette offset',edge,bank),('walls+nuisance',edge,Z),('walls+curvature',np.column_stack((edge,edge[:,:2]*y[:,None]**2,edge[:,2:]*x[:,None]**2)),None)]:
  E=np.asarray(E); n=E.shape[1]; bb=np.r_[base if n>=4 else base[::2],np.zeros(max(0,n-4))]; fs=lambda b,p: pred(A,E,idx,mu,B[:,p].copy(),b[:n])+(N@b[n:])[:,None] if N is not None else pred(A,E,idx,mu,B[:,p].copy(),b)
  if N is not None: bb=np.r_[bb,np.zeros(N.shape[1])]
  scores=[]
  for ps,pts in [([0,1],[0,1]),([0],[1]),([1],[0])]:
   z=least_squares(lambda b:(fs(b,ps)-Y[:,ps]).ravel(),bb,gtol=1e-11).x; scores.append(rms(fs(z,pts)-Y[:,pts]));
   if len(ps)==2: pars=z; resid=fs(z,ps)-Y[:,ps]
  print(tag,np.round(pars,7),scores,'max',np.max(abs(resid)))
  if tag=='walls':
   print('residual shift/bank rms',np.sqrt((resid.reshape(3,8,4,2)**2).mean((1,3)))); print('largest residuals',[(i//len(idx),s['sensor_ij'][i%len(idx)].tolist(),resid[i].round(7).tolist()) for i in np.argsort(abs(resid).max(1))[-12:]])
