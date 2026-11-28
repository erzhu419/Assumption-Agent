from wall_probe import *

def formula(d):
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    lo,hi=k[0],k[-1]
    seam=(lo+hi)/2
    return [(lo+k[1])/(2*lo),0,(seam+h*max(v[0],0))/lo,(hi+k[-2])/(2*hi),0,(seam+h*max(-v[-1],0))/hi]

p=np.array(formula('x')+formula('y'))
for name,q in [('periodic_fvm',p),('open_wall',np.array([1,1,0]*4)),('sealed_wall',np.array([1,0,0]*4))]:
    e=calc(q)-Y
    print(name,'coefficients',q,'pulse RMS',rms(e))
    print('shift-pulse RMS',np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),len(ij),-1)**2,axis=1)))
    print('pulse max abs',np.max(np.abs(e),axis=0))
