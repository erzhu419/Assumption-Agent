from probe import *
from _verified_upstream import _fit,_design
from itertools import product
import json
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=it['study']; s=calibrate_upstream(dict(raw,reference_tracer_pulses=it['campaign_diagnostics'])); sw=raw['sensor_calibration_sweeps']; ref,rr=sw['ref_rosette'],sw['raw_rosette']; co=_fit(ref,rr); print('cal coef',np.round(co,4)); print('sweep error percentiles',np.quantile(np.linalg.norm(_design(ref)@co-rr,axis=1),[0,.25,.5,.75,.9,1])); q=json.load(open('/holdout_task/_upstream_param_payload.json')); print('payload',q); print('sources peak', [np.unravel_index(a.argmax(),a.shape) for a in s['source_modes']]); print('sensors',s['sensor_ij'].T);print('single bank',[(k,np.asarray(v).shape,np.asarray(v).ravel()[:8]) for k,v in sw['single_bank_sweeps'].items()]);
Y=np.array([d['y_pulse_raw'] for d in s['reference_tracer_pulses']]).T; W=np.array([d['w_pulse'] for d in s['reference_tracer_pulses']]).T
Yraw=np.array([d['y_pulse_raw'] for d in it['campaign_diagnostics']]).T; bank=raw['row_bank_id']; Yaff=(Yraw-np.asarray(q['offsets'])[bank,None])/np.asarray(q['gains'])[bank,None]
Z=s['Z_nuisance']; P=np.eye(len(Z))-Z@np.linalg.pinv(Z)

def mat(a,kind):
 k=s['kappa_'+a]; v=s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1); r=s['robin_'+a]; A=axis(k,v,h,[r if kind=='open' else 0,1,r if kind=='open' else 0,1]).tolil()
 if kind in ['periodic','periodic_no_vel','periodic_plus','periodic_local']:
  for i,j,inward in [(0,n-1,1),(n-1,0,n-2)]:
   fi=(k[i]+k[inward])/(2*h*h); fj=(k[i]+k[j])/(2*h*h)
   if kind=='periodic_local': fi=fj=k[i]/h**2
   if kind!='periodic_no_vel':
    fi+=max((1 if i else -1)*v[i],0)/h; fj+=max((-1 if i else 1)*v[i],0)/h
   if kind!='periodic_plus': A[i,inward]=-fi
   A[i,j]=-fj; A[i,i]=-A[i,inward]-A[i,j]
 if kind=='reflect2':
  for i,j in [(0,1),(n-1,n-2)]: A[i,i]*=2; A[i,j]*=2
 return A.tocsc()
nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T; idx=i+nx*j; E=eye(nx*ny); V=diags(s['V_reaction_2d'].ravel(order='F')); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W
res=[]
for x,y in product(['open','closed','periodic','periodic_no_vel','periodic_plus','periodic_local','reflect2'],repeat=2):
 L=kron(eye(ny),mat('x',x))+kron(mat('y',y),eye(nx))+V
 pred=np.concatenate([splu((L+t*E).tocsc()).solve(rhs)[idx] for t in s['mu_shifts']]); e=pred-Y; ea=pred-Yaff
 res.append((np.mean(e**2)**.5,x,y,np.round(np.mean(e**2,axis=0)**.5,6),np.mean((P@e)**2)**.5,np.mean(ea**2)**.5,np.mean((P@ea)**2)**.5))
for k in (0,4,5,6):
 print('sorted column',k); print(*sorted(res,key=lambda x:x[k])[:6],sep='\n')
print('calibrators delta',np.linalg.norm(Y-Yaff,axis=0));print('closed calibrated vs predicted first shift'); pred=response(s,[0,1]*4)@W; print(np.round(np.column_stack((Y[:32],pred[:32],(Y-pred)[:32])),4))
