exec(open('/tmp/verify_thermal.py').read().split('for i,c in enumerate(cases):')[0])
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def forward(k, gamma, sigma, xy, amp):
    n = len(k)
    h = 1/n
    a = lil_matrix((n*n,n*n))
    for i,j in np.ndindex(k.shape):
        p = i*n+j
        for u,v in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
            if 0 <= u < n and 0 <= v < n:
                w = 2/(1/k[i,j]+1/k[u,v])
                a[p,u*n+v] = -w
                a[p,p] += w
            else:
                a[p,p] += gamma*h
    x,y = (np.indices(k.shape)+.5)*h
    q = amp*np.exp(-((x-xy[0])**2+(y-xy[1])**2)/(2*sigma**2))
    temp = spsolve(a.tocsr(),q.ravel()*h*h).reshape(k.shape)
    np.testing.assert_allclose(gamma*h*(temp[0].sum()+temp[-1].sum()+temp[:,0].sum()+temp[:,-1].sum()), q.sum()*h*h, rtol=1e-10)
    return q,temp

limits=np.array([.035,.08,.008])
def check(m,gt):
    t=time.perf_counter(); p=s.localize_thermal_source(m); dt=time.perf_counter()-t
    e=np.array([np.linalg.norm(np.subtract(p['source_xy'],gt['source_xy'])),abs(p['amplitude']/gt['amplitude']-1),abs(p['junction_offset']-gt['junction_offset'])])
    return np.r_[e,dt]

for c in cases:
    m,gt=c['measurement'],c['gt']
    q,temp=forward(m['kappa'],m['gamma_robin'],m['sigma_m'],gt['source_xy'],gt['amplitude'])
    sensor=m['sensor_ij']; exact=temp[tuple(sensor.T)]
    g=s._sensor_green(m['kappa'],m['gamma_robin'],sensor)
    np.testing.assert_allclose(g@q.ravel(),exact,atol=1e-12,rtol=1e-10)
    r=m['T_obs']-exact-gt['junction_offset']
    print('public abs residual quantiles',np.quantile(abs(r),[.5,.8,.9,.94,.96,1]))
    rows=[check({**m,'sensor_ij':np.delete(sensor,i,axis=0),'T_obs':np.delete(m['T_obs'],i)},gt) for i in range(len(sensor))]
    worst=np.max(rows,axis=0); print('public leave-one-sensor-out worst',worst,flush=True)
    assert np.all(worst<=np.r_[limits,15])

rows=[]
for seed in range(36):
    rng=np.random.default_rng(seed)
    n=[15,22,34][seed%3]; gamma=[.65,2.2,5][seed//3%3]; sigma=[.055,.085,.125,.16][seed//9]
    xy=[[.06,.08],[.96,.48],[.43,.76],[.92,.93],[.21,.51],[.47,.05]][seed%6]
    x,y=(np.indices((n,n))+.5)/n
    k=np.exp(.65*np.sin(6*x+2*y)+.6*np.cos(3*x-8*y))
    if seed%2: k=np.where(x>.52,3*k,k)
    amp=18*rng.uniform(.8,1.5)*(.085/sigma)**2; offset=rng.uniform(-.08,.08)
    q,temp=forward(k,gamma,sigma,xy,amp)
    m=[40,52,70][seed//2%3]; sensor=np.column_stack(np.unravel_index(rng.choice(n*n,m,replace=False),(n,n)))
    obs=temp[tuple(sensor.T)]+offset+rng.normal(0,.0006,m)
    bad=rng.choice(m,round(.06*m),replace=False)
    obs[bad]+=rng.uniform(.07,.22,len(bad))*rng.choice([-1,1],len(bad))
    row=check(dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=sensor,T_obs=obs),dict(source_xy=xy,amplitude=amp,junction_offset=offset)); rows.append(row)
    if np.any(row>np.r_[limits,15]): print('stress miss',seed,n,gamma,sigma,xy,row)
worst=np.max(rows,axis=0); print('independent synthetic worst',worst); assert np.all(worst<=np.r_[limits,15])
