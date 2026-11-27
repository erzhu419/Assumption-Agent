from m99_rank2 import *
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
amp=q[:,0]; delta=amp-drift
print('feature rows: x0 x1 x2 x3 drift amp amp-minus-drift'); print(np.c_[G,drift,amp,delta]); print('test telemetry',te['aux_chamber_telemetry'][:6], 'stats',np.mean(te['aux_chamber_telemetry'],0),np.std(te['aux_chamber_telemetry'],0));
for label,V in [('x',G),('x,drift',np.c_[G,drift]),('x,hold',np.c_[G,hc]),('drift,hold',np.c_[drift,hc]),('drift,stress',np.c_[drift,H,gap]),('drift,features,stress',np.c_[drift,G,H,gap])]:
 terms=[()] + [z for k in range(1,4) for z in combinations_with_replacement(range(V.shape[1]),k)]; A=np.column_stack([np.prod(V[:,z],axis=1) for z in terms]); y=delta
 for deg in (1,2,3):
  B=A[:,[len(z)<=deg for z in terms]]; U,s,_=np.linalg.svd(B,full_matrices=False); Q=U[:,s>s[0]*1e-10]; er=y-Q@(Q.T@y); le=(Q*Q).sum(1); print(label,'degree',deg,'rank',len(s[s>s[0]*1e-10]),'fit',np.linalg.norm(er)/len(y)**.5,'loo',np.sqrt(np.mean((er/(1-le))**2)) if np.max(le)<.999999 else None)
 B=A[:,1:]; B=B-B.mean(0); B/=np.linalg.norm(B,axis=0); selected=[]; res=y-y.mean();
 for it in range(5):
  scores=abs(B.T@res); scores[selected]=-1; selected.append(int(np.argmax(scores))); C=np.c_[np.ones(len(y)),A[:,np.array(selected)+1]]; coef=np.linalg.lstsq(C,y,rcond=None)[0]; res=y-C@coef; h=(np.linalg.qr(C)[0]**2).sum(1); print(label,'sparse',it+1,[terms[j+1] for j in selected],coef.round(4),'loo',np.sqrt(np.mean((res/(1-h))**2)).round(6))
