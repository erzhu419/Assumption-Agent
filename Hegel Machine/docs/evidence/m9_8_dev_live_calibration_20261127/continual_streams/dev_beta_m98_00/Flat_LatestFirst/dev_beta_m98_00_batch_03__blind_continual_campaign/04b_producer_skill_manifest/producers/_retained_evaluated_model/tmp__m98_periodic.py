import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_02_solver as base
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; original=base._axis
Y=np.array([base._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); W=np.array([p['w_pulse'] for p in pulses]); Z=s['Z_nuisance']
def axis(k,v,h,robin,kind):
    face,wrap,adv=kind
    A=original(k,v,h,0).tolil()
    for i,j,z in [(0,1,-1),(-1,-2,0)]:
        ki,kj,kz=k[i],k[j],k[z]
        a={'local':ki,'mean':(ki+kj)/2,'harm':2*ki*kj/(ki+kj)}[face]/h**2
        b={'local':ki,'mean':(ki+kz)/2,'harm':2*ki*kz/(ki+kz),'other':kz,'none':0}[wrap]/h**2
        if adv in ('full','inner'):
            a+=max((-1 if i==0 else 1)*v[i],0)/h
        if adv in ('full','wrap'):
            b+=max((1 if i==0 else -1)*v[i],0)/h
        A[i,i]=a+b; A[i,j]=-a; A[i,z]=-b
    return A.tocsc()
ks=list(itertools.product(('local','mean','harm'),('local','mean','harm','other','none'),('no','full','wrap','inner')))
by_pulse=[[],[]]
for kind in ks:
    base._axis=lambda k,v,h,r:axis(k,v,h,r,kind)
    S,U=base._design(s); R=W@S.T-Y
    rms=np.sqrt(np.mean(R**2,axis=1)); proj=R-(R@np.linalg.pinv(Z).T)@Z.T
    for j in range(2): by_pulse[j].append((rms[j],kind,rms,np.sqrt(np.mean(proj**2,axis=1))))
base._axis=original
for j,b in enumerate(by_pulse):
 print('train pulse',j)
 for row in sorted(b,key=lambda a:a[0])[:8]: print(row)
print('grid',s['nx'],s['ny'],'robin',s['robin_x'],s['robin_y'],'mu',s['mu_shifts'])
