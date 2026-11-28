exec(open('/tmp/validate_batch04.py').read().split('records=[]')[0])
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

def predict(train_idx, test_idx, sets):
    tt=np.asarray(s['eval_times']); tr=t[train_idx]; g=[b[train_idx][:,k] for k in sets]; axes=[b[test_idx][:,k] for k in sets]
    def design(r,g):
        h=-np.expm1(-tt[:,None]*r)
        return np.concatenate([(v[:,:,None]*h[None,None,:,j]).transpose(0,2,1).reshape(len(v)*len(tt),v.shape[1]) for j,v in enumerate(g)],axis=1)
    def solve(r):
        d=design(r,g); c=m._ls(d,tr.ravel()); return d,c
    def resid(r):
        d,c=solve(r); return d@c-tr.ravel()
    r=least_squares(resid,[1.81,.42],bounds=([.8,.04],[5.,.75]),ftol=1e-10,gtol=1e-10,xtol=1e-10).x
    d,c=solve(r)
    return (design(r,axes)@c).reshape(len(test_idx),len(tt)),r,c,abs(d@c-tr.ravel()).mean()

fast=[1,2,3,4,9,14,19,24]; slow=[5,6,7,8,10,15,20,21]
with threadpool_limits(limits=1):
 for label,sets in [('full',[list(range(25))]*2),('branches',[fast,slow]),('branches-intercept',[[0]+fast,[0]+slow]),('cross-both',[list(range(1,5))+fast[4:]+slow[4:],list(range(5,9))+fast[4:]+slow[4:]])]:
    rec=[]
    for name,i,j in splits:
        pred,r,c,trainerr=predict(i,j,sets); et=abs(pred-t[j]); rec.append((name,et.mean(),et.mean(0).max(),et.max(), r, trainerr))
    print(label,'official',rec[0],'worst aggregate and timestamp',max(rec,key=lambda w:w[1]),max(rec,key=lambda w:w[2])); print('full fit',predict(np.arange(72),np.arange(72),sets)[1:])
