exec(open('/tmp/stress_node00.py').read().split('for k in families:')[0])
from scipy.optimize import least_squares
from pathlib import Path

def build(x,q,kind):
    a=q[0]; b=q[1] if kind=='exponent' else .5
    sat=x/(1+a*x*x)**b
    if kind=='linear': sat=sat+q[1]*x
    return np.c_[np.ones(len(x)),terms(x),sat]

def fitted(x,y,kind):
    p0=[.85]; lower=[.05]; upper=[6.]
    if kind=='linear': p0+=[0.]; lower+=[-.25]; upper+=[.25]
    if kind=='exponent': p0+=[.5]; lower+=[.15]; upper+=[1.5]
    def residual(q):
        f=build(x,q,kind); return (f@np.linalg.lstsq(f,y,rcond=1e-9)[0]-y).ravel()
    q=least_squares(residual,p0,bounds=(lower,upper),gtol=1e-13).x
    c=np.linalg.lstsq(build(x,q,kind),y,rcond=1e-9)[0]
    return q,lambda z:build(z,q,kind)@c
for kind in ('pure','linear','exponent'):
    q,pred=fitted(u,y,kind); re=[]
    for i in range(len(u)):
        m=np.arange(len(u))!=i
        re.extend((fitted(u[m],y[m],kind)[1](u[~m])-y[~m]).ravel())
    print(kind,'parameters',q,'fit RMSE',np.sqrt(np.mean((pred(u)-y)**2)),'LOO RMSE',np.sqrt(np.mean(np.square(re))),'worst absolute panel error',max(abs(np.array(re))))
path=Path('/holdout_task/node_00.py'); src=path.read_text(); a=src.index('    def curvature('); b=src.index('    def fit(',a)
compact='''    def curvature(w, q):
        # Cyclic nearest-neighbor products and shared odd saturation.
        cross = w * np.roll(w, -1, axis=1)
        saturation = w / np.sqrt(1 + q[0]*w*w)
        return np.c_[np.ones(len(w)), cross, saturation]

'''
candidate=src[:a]+compact+src[b:]; candidate=candidate.replace('[1.2], (.2, 4.)','[.85], (.05, 6.)'); Path('/tmp/saturation_node00.py').write_text(candidate)
