import pathlib
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from thermal_sink_solver import _thermal_operator, _mad
src=pathlib.Path('/tmp/thermal_validate.py').read_text()
exec(src.split('cases = pickle.load',1)[0])
for c in pickle.load(open('/holdout_task/public_thermal_cases.pkl','rb')):
    m,g=c['measurement'],c['gt']; n=m['N']; h=1/n; axis=(np.arange(n)+.5)*h
    q=g['amplitude']*h*h*np.exp(-((axis[:,None]-g['source_xy'][0])**2+(axis[None,:]-g['source_xy'][1])**2)/(2*m['sigma_m']**2))
    u=spsolve(_thermal_operator(m['kappa'],m['gamma_robin']),q.ravel()).reshape(n,n)
    i,j=m['sensor_ij'].T; y=m['T_obs']; clean=u[i,j]+g['junction_offset']; bad=np.abs(y-clean)>.02
    print('public gross sensors: observed, clean, difference',np.column_stack((y,clean,y-clean))[bad])

def compare(m,g,bad):
    n=m['N']; i,j=m['sensor_ij'].T; ids=i*n+j; obs=m['T_obs']; sigma=m['sigma_m']; kappa=m['kappa']
    x=(np.arange(n)+.5)/n; X,Y=np.meshgrid(x,x,indexing='ij'); coords=np.column_stack([X.ravel(),Y.ravel()])
    eye=np.zeros((n*n,len(ids))); eye[ids,np.arange(len(ids))]=1
    G=splu(_thermal_operator(kappa,m['gamma_robin'])).solve(eye).T
    def predict(z):
        d=coords-z[:2]; q=np.exp(-.5*np.sum(d*d,axis=1)/sigma**2)/n**2
        b=np.einsum('ij,j->i',G,q); J=np.column_stack((np.einsum('ij,jk->ik',G,q[:,None]*d/sigma**2)*z[2],b,np.ones(len(b))))
        return z[2]*b+z[3],J
    def vec(p): return np.r_[p['source_xy'],p['amplitude'],p['junction_offset']]
    def unpack(z): return dict(source_xy=z[:2].tolist(),amplitude=float(z[2]),junction_offset=float(z[3]))
    z=vec(localize_thermal_source(m)); t=vec(g); keep=np.ones(len(obs),bool); keep[bad]=False
    residual=lambda z:(predict(z)[0]-obs)[keep]; jac=lambda z:predict(z)[1][keep]
    kwargs=dict(jac=jac,bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]),gtol=1e-12)
    o=least_squares(residual,t,**kwargs).x
    a=least_squares(residual,z,**kwargs).x
    J=jac(t); err=np.array([errors(unpack(v),g) for v in (z,o,a)]); rr=predict(z)[0]-obs; est=np.abs(rr-np.median(rr))<=3.8*_mad(rr)
    sd=np.sqrt(np.diag(np.linalg.pinv(J.T@J)))*.0013
    return err, [int(np.sum(est[bad])),int(np.sum(~est[keep]))], sd, np.linalg.norm(residual(o))

part=src[src.index('def truth_grid'):]
part=part.replace('results.append((seed,run(dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack((i,j)),T_obs=obs),dict(source_xy=xy,amplitude=amp,junction_offset=offset))))','''d=dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack((i,j)),T_obs=obs)
        g=dict(source_xy=xy,amplitude=amp,junction_offset=offset)
        r, masks, sd, fit=compare(d,g,bad)
        results.append(r)
        if np.any(r[0]>LIMIT): print(name,seed,'baseline/oracle/oracle-from-fit',np.round(r,4).tolist(),'gross retained / clean removed',masks,'linearized sd',np.round(sd,4),'oracle norm',round(fit,4))''').replace('report(name,results)',"print(name,'oracle maxima',np.max(results,axis=0),'misses',np.sum(np.asarray(results)>LIMIT,axis=0),flush=True)")
exec(part)
