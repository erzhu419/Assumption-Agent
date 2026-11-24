import pickle
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit
D=pickle.load(open('/holdout_task/public_screen.pkl','rb'))
P=D['plates']; truth=D['gt']

def coordinates(wells):
    return np.array([(w['row'],w['col']) for w in wells],float).T

def basis(rc,kind,t):
    r,c=rc
    a,b=r/15,c/23
    dr,dc=np.minimum(r,15-r),np.minimum(c,23-c)
    d=np.minimum(dr,dc)
    one=np.ones_like(r)
    if kind=='scalar': return one[:,None]
    if kind=='poly': return np.array([one,a,b,(a-.5)**2,(b-.5)**2,(a-.5)*(b-.5)]).T
    er,ec=np.exp(-dr/t),np.exp(-dc/t)
    em=np.exp(-d/t)
    ss=np.exp(-r/t)+np.exp(-(15-r)/t)+np.exp(-c/t)+np.exp(-(23-c)/t)
    extras={'min': [em], 'min_rc':[em,a,b], 'sum_rc':[ss,a,b], 'split_rc':[er,ec,a,b], 'corner_rc':[er,ec,er*ec,a,b], 'min_corner_rc':[em,er*ec,a,b], 'min_cross':[em,a,b,a*b], 'min_quad':[em,a,b,(a-.5)**2,(b-.5)**2]}[kind]
    return np.array([one]+extras).T

def compute(p,kind,t,keep):
    w=p['wells']; rc=coordinates(w); y=np.array([q['fluorescence'] for q in w]); ty=np.array([q['well_type'] for q in w]); x=basis(rc,kind,t)
    bg=np.median(y[ty=='POS_KILL']); sel=(ty=='DMSO') & keep
    X=x[sel]; Y=y[sel]-bg; scale=np.median(Y)
    z=np.linalg.lstsq(X,Y/scale,rcond=None)[0]
    fit=least_squares(lambda z:X@z-Y/scale,z,loss='cauchy',f_scale=.035)
    gain=x@fit.x*scale
    return (y-bg)/gain, gain, y-bg, ty

def get_keep(p,fold):
    w=p['wells']; rc=coordinates(w)
    return ((rc[0]*2+rc[1]//3)%4 != fold) if fold is not None else np.ones(len(w),bool)

def fitcurve(x,y,g):
    start=x[np.argmin(np.abs(y-.5))]; w=g/np.median(g)
    f=lambda z: (expit(np.log(10)*z[1]*(z[0]-x))-y)*w
    return least_squares(f,[start,1.],bounds=([x.min()-4,.1],[x.max()+4,8]),loss='cauchy',f_scale=.035).x

def run(kind,t,fold=None,exclude=None):
    controls=[]; rows=defaultdict(list)
    for j,p in enumerate(P):
        keep=get_keep(p,fold); v,g,y,ty=compute(p,kind,t,keep)
        test=(ty=='DMSO') & (~keep if fold is not None else keep)
        r,c=coordinates(p['wells']); depth=np.minimum.reduce([r,15-r,c,23-c]); controls.extend(zip((g[test]-y[test])/np.median(y[ty=='DMSO']),depth[test]))
        if j==exclude: continue
        for i,q in enumerate(p['wells']):
            if ty[i]=='COMPOUND': rows[q['compound_id']].append((q['log10_conc_uM'],v[i],g[i]))
    return np.array(controls),{k: fitcurve(*np.array(a).T) for k,a in rows.items()}

def errors(out):
    e={k:abs(v-np.array([truth[k]['log10_ic50_uM'],truth[k]['hill_slope']])) for k,v in out.items()}
    return np.max(list(e.values()),axis=0), [max(e,key=lambda k:e[k][i]) for i in range(2)]

if __name__=='__main__':
    for p in P[:1]:
        print('DMSO positions:',[(w['row'],w['col']) for w in p['wells'] if w['well_type']=='DMSO'])
    for kind in ['scalar','poly','min','min_rc','sum_rc','split_rc','corner_rc','min_corner_rc','min_cross','min_quad']:
        def score(t):
            acc=[]
            for p in P:
                for f in range(4):
                    keep=get_keep(p,f); _,g,y,ty=compute(p,kind,t,keep); dm=ty=='DMSO'; e=(g[dm&~keep]-y[dm&~keep])/np.median(y[dm&keep]); acc.extend(e)
            return np.sqrt(np.mean(np.square(acc)))
        t=minimize_scalar(score,bounds=(.65,10),method='bounded',options={'xatol':.02}).x if kind not in ('scalar','poly') else 3.
        co,out=run(kind,t); jack=np.max([errors(run(kind,t,exclude=j)[1])[0] for j in range(4)],axis=0)
        print(kind, f't={t:.3f} cv={score(t):.4f}', 'public',errors(out),'drop plate',np.round(jack,4))
