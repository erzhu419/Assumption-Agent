import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x0=c['ref_rosette']; y0=c['raw_rosette']; good=np.max(abs(y0),axis=1)<5; x=x0[good]; y=y0[good]
print('Bank scalar families: model, max residual for banks, parameters for bank 0')
for f,n in [(lambda x,p:p[0]+p[1]*x+p[2]*np.tanh(p[3]*x),4),(lambda x,p:p[0]+p[1]*x+p[2]*np.arctan(p[3]*x),4),(lambda x,p:p[0]+p[1]*x+p[2]*x/(1+p[3]*x*x),4),(lambda x,p:p[0]+p[1]*x+p[2]*x/np.sqrt(1+p[3]*x*x),4),(lambda x,p:p[0]+p[1]*x+p[2]*np.tanh(p[3]*x+p[4]),5)]:
 scores=[]; params=[]
 for j in range(4):
  b=c['single_bank_sweeps']; a=(b['bank_id']==j)&(abs(b['raw_readout'])<5); t,z=b['ref_signal'][a],b['raw_readout'][a]; r=least_squares(lambda p:f(t,p)-z,[0,.1,1,1]+[0]*(n-4),bounds=([-3,-5,-5,.03]+[-2]*(n-4),[3,5,5,30]+[2]*(n-4))); scores.append(max(abs(r.fun)));params.append(r.x)
 print(np.round(scores,7),np.round(params[0],7))
print('per-output additive polynomial LOOCV by degree (all-channels vs diagonal+paired cross)')
paired=[2,3,0,1]
for deg in range(1,11):
 row=[]
 for cols in (lambda j:range(4),lambda j:(j,paired[j])):
  pred=[]
  for j in range(4):
   A=np.column_stack([np.ones(len(x))]+[x[:,k]**d for k in cols(j) for d in range(1,deg+1)]); Q,R=np.linalg.qr(A,mode='reduced'); e=(y[:,j]-Q@(Q.T@y[:,j]))/(1-np.sum(Q*Q,axis=1)); pred.append(e)
  e=np.array(pred).T; row.append(np.round([np.sqrt(np.mean(e*e)),np.max(abs(e)),np.sqrt(np.mean(e[np.max(abs(x),axis=1)<=1.25]**2))],7))
 print(deg,*row)
print('paired fit residual correlations with all quadratic interactions:')
for j,k in enumerate(paired):
 A=np.column_stack([np.ones(len(x))]+[x[:,i]**d for i in (j,k) for d in range(1,8)]); e=y[:,j]-A@np.linalg.lstsq(A,y[:,j],rcond=None)[0]; terms=np.array([x[:,u]*x[:,v] for u in range(4) for v in range(u+1,4)]); print(j,np.round(np.corrcoef(np.vstack((e,terms)))[0,1:],3),np.std(e))
