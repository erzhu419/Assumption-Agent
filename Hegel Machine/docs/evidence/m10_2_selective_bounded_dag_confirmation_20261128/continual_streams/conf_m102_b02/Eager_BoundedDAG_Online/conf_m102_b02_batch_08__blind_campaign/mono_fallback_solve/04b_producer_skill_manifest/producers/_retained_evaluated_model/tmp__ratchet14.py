import sys, pickle
sys.path[:0]=['/holdout_task','/tmp']
from batch_solver import *
import low_rank_head as h
for k in ('D','P','fit','np'):setattr(h,k,globals()[k])
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); ii=np.arange(n); folds=[(ii[:52],ii[52:])]+[(np.setdiff1d(ii,j),j) for j in np.array_split(np.random.default_rng(88).permutation(n),4)]
cache=[]
for i,j in folds:
    z,b,e,C,W=features(s,t['x'][np.r_[i,j]]); y=t['y'][i]; yh=t['y'][j]; opt=_scalar_candidates(z,b,e,C,W,y) if '_scalar_candidates' in globals() else None
    if opt is None:
        import batch_solver as solver
        opt=solver._scalar_candidates(z,b,e,C,W,y)
    base=.5*(opt['original']+opt['blend30']); cache.append((z,b,C,y,yh,base))
    sh=kernel(np.c_[D(P(.5*b))@C,.22*z],len(i),2,0,.0005,t['traj'][i]); err=np.abs(sh-t['traj'][j]); print('state average/worst horizon',err.mean(),err.mean(0).max(),flush=True)
base_err=np.array([np.mean(abs(a[-1]-a[-2])) for a in cache]); best=((base_err.mean()+base_err.max())/2,base_err,None,0.)
for name,ps in [(k,(.055,.19,.5)) for k in ('rank1','rank2','fullrank1','fullrank2')]+[(f'proj{k}{m}',(.10,1.2)) for k in ('sin','asinh','tanh','rat','sq') for m in ('p','c')]:
    for pen in ps:
        try: preds=[h._structured_y(*a[:4],name,pen) for a in cache]
        except (ValueError,np.linalg.LinAlgError):continue
        for w in (1.,.75,.5,.25):
            es=np.array([np.mean(abs(w*p+(1-w)*a[-1]-a[-2])) for p,a in zip(preds,cache)]); score=(es.mean()+es.max())/2
            if es[0] <= .038:score-=1
            if score<best[0] and es[0]<=base_err[0]:best=(score,es,(name,pen),w)
print('CHOSEN',best,flush=True)
if best[2] is not None:
    p='/holdout_task/batch_solver.py'; src=open(p).read(); a,w=best[2],best[3]; old="    y=0.5*options['original']+0.5*options['blend30']"; new=f"    y={1-w!r}*(0.5*options['original']+0.5*options['blend30'])+{w!r}*_structured_y(z,b,C,tr['y'],{a[0]!r},{a[1]!r})"; assert old in src
    open(p,'w').write(src.replace(old,new)+'\n'+open('/tmp/low_rank_head.py').read())
