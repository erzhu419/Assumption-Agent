import pickle
import numpy as np
import tomography_solver as m
for k,item in enumerate(pickle.load(open('/holdout_task/public_surveys.pkl','rb'))):
    s,gt=item['survey'],item['gt']; obs=np.asarray(s['T_obs_s']); _,_,e=m._fit(s); e=e.reshape(obs.shape)
    bad=abs(e)>2; base=obs.copy(); base[bad]-=np.median(abs(e[bad]))*np.sign(e[bad])/1000
    n=int(bad.sum())
    for axis in (0,1):
        results=[]
        for j in range(obs.shape[axis]):
            for end in (0,1):
                for sign in (-1,1):
                    y=base.copy(); selected=slice(-n,None) if end else slice(n)
                    if axis==0: y[j,selected]+=sign*.006
                    else: y[selected,j]+=sign*.006
                    r=m.invert_plume_and_clocks(dict(s,T_obs_s=y))
                    a=np.linalg.norm(np.subtract(r['plume_xz_m'],gt['plume_xz_m'])); b=np.sqrt(np.mean(np.square(np.subtract(r['rcv_clock_bias_s'],gt['rcv_clock_bias_s']))))
                    results.append((a,b,j,end,sign))
        print(k,axis,'worst location',max(results),'worst clock',max(results,key=lambda r:r[1]),'failures',sum(a>1.8 or b>.00025 for a,b,*_ in results),'/',len(results),flush=True)
