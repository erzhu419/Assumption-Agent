import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
import compare_chamber as chamber

path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'batch_solver.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s=pickle.load(open(path+'public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']
Xs=[]; Ys=[]
for bank in range(4):
    ids=b['bank_id']==bank
    v=np.zeros((sum(ids),4)); v[:,bank]=b['ref_signal'][ids]
    Xs.append(np.vstack((x,v)))
    Ys.append(np.r_[y[:,bank],b['raw_readout'][ids]])
X=np.array(Xs); Y=np.array(Ys)
A=np.array([m._rosette_features(X[i],i) for i in range(4)])
a=A[:,:,1]; nxt=A[:,:,2]
# Added squared sums/differences span new pairwise quadratic directions;
# the existing a*b term already spans their difference.
Ap=np.concatenate((A,((a+nxt)**2)[...,None],((a-nxt)**2)[...,None]),axis=2)

def fit(A, train):
    co=[]; masks=[]
    for D,target,k in zip(A,Y,train):
        keep=k.copy()
        for _ in range(12):
            c=np.linalg.lstsq(D[keep],target[keep],rcond=None)[0]
            err=abs(D@c-target)
            new=k & (err<=max(6*np.median(err[k]),1e-10))
            if np.array_equal(new,keep): break
            keep=new
        co.append(np.linalg.lstsq(D[keep],target[keep],rcond=None)[0]); masks.append(keep)
    return np.array(co),np.array(masks)

pred={k:np.empty_like(Y) for k in ['cubic','paired','shared','rational-own','rational-sum','rational-difference']}
for k in range(8):
    test=np.broadcast_to(np.arange(Y.shape[1])[None,:]//3%8==k,Y.shape)
    co,keep=fit(A,~test); pc,_=fit(Ap,~test)
    pred['cubic'][test]=np.einsum('bni,bi->bn',A,co)[test]
    pred['paired'][test]=np.einsum('bni,bi->bn',Ap,pc)[test]
    p0=np.r_[co[:,0],co[:,1],np.mean(co[:,2:]/co[:,1,None],axis=0)]
    def shared(p): return p[:4,None]+p[4:8,None]*(a+A[:,:,2:]@p[8:])
    par=least_squares(lambda p:(shared(p)-Y)[keep],p0,gtol=1e-13).x
    pred['shared'][test]=shared(par)[test]
    for name,q in [('own',a),('sum',a+nxt),('difference',a-nxt)]:
        def rat(p): return np.einsum('bni,bi->bn',A,p[:-1].reshape(4,6))/(1+p[-1]*q*q)
        par=least_squares(lambda p:(rat(p)-Y)[keep],np.r_[co.ravel(),1e-6],bounds=(np.r_[np.full(24,-np.inf),0.],np.inf),gtol=1e-12).x
        pred['rational-'+name][test]=rat(par)[test]
# Common gross-outlier mask from entirely out-of-fold baseline predictions.
inlier=abs(pred['cubic']-Y)<.01
print('gross calibration points by bank',(~inlier).sum(axis=1).tolist())
for name,P in pred.items():
    e=P-Y
    print(name,'clean RMSE',np.sqrt(np.mean(e[inlier]**2)),'worst bank RMSE',max(np.sqrt(np.mean(r[k]**2)) for r,k in zip(e,inlier)),'max clean error',abs(e[inlier]).max())

chamber.y=m._calibrate(s,s['y_obs_raw']); iy=np.arange(chamber.y.size)
S,_=m._responses(s)
per,_=chamber.m._responses(s)
rows=len(iy)//len(s['mu_shifts'])
for name,masklist in [('shift',[iy//rows==j for j in range(len(s['mu_shifts']))]),('sensor rosette',[iy%rows//4==j for j in range(rows//4)]),('bank',[iy%4==j for j in range(4)])]:
    for title,design in [('Robin',S),('periodic',per)]:
        scores=[]; bias=[]; unprofiled=[]
        for test in masklist:
            err,w=chamber.evaluate(design,~test,s['source_ridge'])
            scores.append(np.sqrt(np.mean(err[test]**2)))
            bias.append(abs(np.mean(err[test])))
            unprofiled.append(np.sqrt(np.mean((chamber.y-design@w)[test]**2)))
        print(name,title,'slice RMSE',np.round(scores,6),'worst abs mean',max(bias),'worst before applying fitted nuisance',max(unprofiled))
