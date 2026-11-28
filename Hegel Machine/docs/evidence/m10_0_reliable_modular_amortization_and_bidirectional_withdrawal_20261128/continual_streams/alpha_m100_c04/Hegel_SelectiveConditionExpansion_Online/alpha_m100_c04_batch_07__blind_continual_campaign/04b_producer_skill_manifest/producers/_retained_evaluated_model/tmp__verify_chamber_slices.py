import sys,json,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import downstream_solver as d
import batch_solver as b
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; payload=json.load(open("/holdout_task/_upstream_param_payload.json")); cal=calibrate_upstream(raw); out=b.solve_batch(raw)
for k,v in d.solve_downstream(cal).items():
    np.testing.assert_array_equal(v,out[k]); assert np.all(np.isfinite(v)),k
u,w=out['estimated_taus'],out['hereditary_weights']; assert .05<u[0]<u[1]<25
tr,te=cal['train'],cal['test']; dt,t=map(np.asarray,[cal['past_dt'],cal['eval_times']]); x=d._features(tr,te)[len(tr['lab_id']):]; inp=d._inputs(te,x); past,hold,c=inp
z=np.zeros((len(hold),2))
for s,h in enumerate(dt): z=np.exp(-h/u)*z-np.expm1(-h/u)*past[:,s,None]
np.testing.assert_allclose(d._design(u,inp,dt,t)[1][:,:2]*u,hold[:,None]-z,atol=2e-14)
f0,r0=d._chamber_response(u,w,inp,dt,np.array([0.,1e-7])); np.testing.assert_allclose(f0[:,0],0.,atol=1e-14); np.testing.assert_allclose(f0[:,1]/1e-7,r0,rtol=3e-7,atol=1e-8)
print('Material taus and weights',u,w,'max output difference versus staged bridge payload', {k:float(np.max(abs(v-out[k]))) for k,v in d.solve_downstream(calibrate_upstream(raw,payload)).items()})
g=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),payload)['test']; n=len(g['lab_id']); xg=d._features(g,g)[:n]; ig=d._inputs(g,xg); lab=g['lab_id']; complete=np.isfinite(g['coupon_features_raw']).all(axis=1)
a,h=d._design(u,ig,dt,t); y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; base=np.c_[a[:,:,:2]@w[:2],h[:,:2]@w[:2]]; full=np.c_[a@w,h@w]; M=np.r_[np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1][None,:]]; J=(ig[2]@w[2:])[:,None,None]*M
initial=M[-1]; origin=np.array([0.,0.,1.]); null=np.vstack([np.eye(2),-initial[:2]])
def fit(mask,constrained):
    A=J[mask].reshape(-1,3); rhs=(y-base)[mask].ravel()
    if constrained: return origin+null@np.linalg.lstsq(A@null,rhs-A@origin,rcond=None)[0]
    return np.linalg.lstsq(A,rhs,rcond=None)[0]
k=fit(complete,False); np.testing.assert_allclose(fit(complete,True)[:2],d._CHAMBER_COUPLING,atol=3e-8,rtol=0); print('coupling; initial driver multiplier',k,k@initial,'constrained',fit(complete,True))
p=np.empty_like(y); q=np.empty_like(y)
for l in np.unique(lab):
    v=lab==l; train=~v&complete
    p[v]=base[v]+J[v]@fit(train,False); q[v]=base[v]+J[v]@fit(train,True)
def metric(pred,v):
    e=(pred-y)[v]; return np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,:len(t)//2]**2),np.mean(e[:,len(t)//2:-1]**2),np.mean(e[:,-1]**2)])
slices={**{f'lab-{l}':lab==l for l in np.unique(lab)},'complete':complete,'missing':~complete}
results=[]
for name,v in slices.items():
    row=np.array([metric(pred,v) for pred in [p,q,full]]); results.append(row)
    print(name,'n',v.sum(),'RMSE strain/early/late/rate: free, initial-rate preserving, baseline',np.round(row,6).tolist())
    assert np.all(row[1,:3]<row[2,:3]),name
    np.testing.assert_allclose(row[1,3],row[2,3],atol=1e-13,rtol=0)
print('worst slices',np.round(np.max(results,axis=0),6).tolist(),'pooled',np.round([metric(pred,np.ones(n,bool)) for pred in [p,q,full]],6).tolist())

inc, rate = d._chamber_response(u,w,ig,dt,t)
np.testing.assert_allclose(np.c_[inc,rate],base+J@fit(complete,True),atol=3e-8,rtol=0)
