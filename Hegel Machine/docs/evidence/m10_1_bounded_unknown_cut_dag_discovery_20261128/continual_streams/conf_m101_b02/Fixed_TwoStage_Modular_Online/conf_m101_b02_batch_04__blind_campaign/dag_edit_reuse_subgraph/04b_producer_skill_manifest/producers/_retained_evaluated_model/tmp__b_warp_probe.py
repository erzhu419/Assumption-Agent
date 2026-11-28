exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
from scipy.optimize import least_squares
u,v=s['panel_b']['u'],s['panel_b']['v']

def warp(a,p):
    return a+p[1]*np.sin(p[0]*a)+(p[2]*a*a if len(p)>2 else 0)

def profiled(ix,p):
    d=m._affine(warp(u,p)); return d@m._fit(d[ix],v[ix])-v

ids=np.arange(len(u)); errs={}
for label, start in [('shared_sine',[2.83,-.4]),('shared_quadratic_sine',[2.83,-.4,0.])]:
    r=[]
    for i in [-1,*ids]:
        ix=ids[ids!=i]; p=least_squares(lambda p:profiled(ix,p)[ix].ravel(),start).x
        if i<0: print(label,'parameters',p.round(6),'fit RMSE',np.sqrt(np.mean(profiled(ix,p)**2)))
        else: r.append(profiled(ix,p)[i])
    errs[label]=np.sqrt(np.mean(np.square(r)))
print('refit leave-one-row-out RMSE',errs,'versus 0.008422 for current panel B')
