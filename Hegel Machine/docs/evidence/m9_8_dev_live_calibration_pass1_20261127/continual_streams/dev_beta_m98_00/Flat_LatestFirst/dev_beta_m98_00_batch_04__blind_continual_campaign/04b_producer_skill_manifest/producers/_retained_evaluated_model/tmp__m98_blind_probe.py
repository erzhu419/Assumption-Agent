import runpy,pickle,numpy as np
from scipy.optimize import least_squares,lsq_linear
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
z=s['Z_nuisance']; src=s['source_modes']; nominal=np.ones(8); S0,U0=m['_response'](s,nominal,src); y0=m['_calibration'](s['sensor_calibration_sweeps'])(s['y_obs_raw']); n=len(y0); budget=s['total_budget']; cap=s['u_box']; rows=np.arange(n); folds=(rows//4)%4; w0=m['solve_batch'](s)['weights']; e=y0-S0@w0; gamma=np.linalg.lstsq(z,e,rcond=None)[0]; e-=z@gamma

def fit(y, mask, groups):
    def response(x):
        w=nominal.copy()
        for ids,v in zip(groups,x): w[ids]=np.exp(v)
        return m['_response'](s,w,src)
    def profile(x):
        S,U=response(x); A=np.c_[S,z]; a=A[mask]; v=y[mask]; q=np.linalg.lstsq(np.vstack([a,np.r_[np.ones(8),np.zeros(3)]*10]),np.r_[v,10*budget],rcond=None)[0]
        if np.any((q[:8]<0)|(q[:8]>cap)):
            q=lsq_linear(np.vstack([a,np.r_[np.ones(8),np.zeros(3)]*10]),np.r_[v,10*budget],bounds=(np.r_[np.zeros(8),np.full(3,-np.inf)],np.r_[np.full(8,cap),np.full(3,np.inf)]),tol=2e-6).x
        return A@q-y,S,U
    x=np.zeros(len(groups))
    if groups: x=least_squares(lambda x: np.r_[profile(x)[0][mask],.007*x],x,bounds=(-1.5,1.5),max_nfev=18,diff_step=.002).x
    return x,profile(x)

def test(tag,w,y):
    S,U=m['_response'](s,w,src); P=lambda x:x-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@x); obj=lambda A:m['_weights'](P(A),P(y),s['source_ridge'],cap,budget); f=np.tensordot(obj(S),U,axes=1)
    for g in ([],[[0,1],[4,5]],[[0],[1],[4],[5]]):
        errors=[]
        for i in range(4):
            tr=folds!=i; _, (r,*_)=fit(y,tr,g); errors.append(np.sqrt(np.mean(r[~tr]**2)))
        x,(_,A,B)=fit(y,rows>=0,g); pred=np.tensordot(obj(A),B,axes=1); print(tag,len(g),'CV',np.round(errors,5),'parameters',np.round(np.exp(x),3),'field err',round(np.linalg.norm(pred-f)/np.linalg.norm(f),4),flush=True)

test('real public',nominal,y0)
for tag,w in [('axis',np.array([.55,.55,1,1,1.6,1.6,1,1])),('side',np.array([.4,1.8,1,1,.75,1.4,1,1]))]:
    S,_=m['_response'](s,w,src); test(tag,w,S@w0+z@gamma+e)
