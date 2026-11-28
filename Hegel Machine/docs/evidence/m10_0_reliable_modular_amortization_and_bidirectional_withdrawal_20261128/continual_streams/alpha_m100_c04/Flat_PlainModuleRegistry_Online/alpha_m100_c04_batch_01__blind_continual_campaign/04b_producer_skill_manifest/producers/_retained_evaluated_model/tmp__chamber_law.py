exec(open('/tmp/chamber_probe.py').read().split('for name,y')[0])
from scipy.linalg import lstsq
n=len(hold); lab=v['lab_id']; drift=cov@w[2:]; P=np.c_[np.ones(n),hold,past]; F=np.c_[np.ones(n),x]; np.set_printoptions(precision=5)
for C in [P,F,np.c_[P,x],np.c_[P,drift],np.c_[P,drift,hold*drift]]:
 print('design columns',C.shape[1]); pred=np.empty_like(Y)
 for L in np.unique(lab):
  j=lab!=L; A=basis(tau); b=lstsq(C[j],Y[j])[0]; b=lstsq(A,b.T)[0].T; pred[~j]=C[~j]@b@A.T
 print('LOLO curve RMSE,rate RMSE',[(L,np.mean((Y[lab==L,:-1]-pred[lab==L,:-1])**2)**.5,np.mean((Y[lab==L,-1]-pred[lab==L,-1])**2)**.5) for L in np.unique(lab)])
 b=lstsq(C,Y)[0]; print('amp coefficients',lstsq(basis(tau),b.T)[0].T)
print('amp drift vs intrinsic drift',np.c_[drift,fit(tau,Y)[1][:,2]])
for k in ['train','test']:
 Pk,Hk,_=m._inputs(d[k],m._features(d['train'],d['test'])[:72] if k=='train' else m._features(d['train'],d['test'])[72:],m._calibration(d['calibration_load_cells'])); print(k,'hold',np.quantile(Hk,[0,.1,.5,.9,1]),'past min max',Pk.min(0),Pk.max(0))
print('diag','hold',np.quantile(hold,[0,.1,.5,.9,1]),'past min max',past.min(0),past.max(0));print('past samples',past[:8]);print('diag labs & strata',np.c_[lab,v['stratum']]);print('diagnostic feature normalization alternatives')
for p0,p1,i in [(v,v,0),(v,d['train'],0),(v,d['test'],0)]:
 xx=m._features(p0,p1)[:n]; dd=w[2]+xx@w[3:]; c=np.c_[np.ones(n),dd]; target=fit(tau,Y)[1][:,2]; print(lstsq(c,target)[0],np.std(c@lstsq(c,target)[0]-target))
