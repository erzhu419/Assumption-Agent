exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from pathlib import Path
from scipy.optimize import least_squares
p=Path('/holdout_task/node_01.py'); original=p.read_text(); helpers=Path('/tmp/label_final_helpers.txt').read_text(); exec(original+'\n'+helpers)
t=Y[:,0]; n=len(t)
allfold=fs + [(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(46).permutation(n),4)]
# Preserve agreement with the observed paired coefficients; fit the remaining label component separately.
def nonlinear(u,z,t,spec):
    ib, dims, kind = spec
    n=len(t); a=_label_features(u,z)[0][ib]; q,r=np.linalg.qr(a[:n]); b=t-q@(q.T@t)
    x={'z':z,'u':u,'uz':np.c_[u,z],'w':u*np.roll(z,1,axis=1),'zw':np.c_[z,u*np.roll(z,1,axis=1)]}[dims]
    f={'sin':np.sin,'tanh':np.tanh,'square':lambda v:v*v,'rational':lambda v:1/(1+v*v),'cube':lambda v:v**3}[kind]
    def design(p):return f(p[0]+x@p[1:])
    def res(p):
        v=design(p)[:n]; v=v-q@(q.T@v); return b-v*(v@b)/max(v@v,1e-12)
    rng=np.random.default_rng(123)
    starts=np.r_[rng.normal(0,.7,(5,x.shape[1]+1)),np.c_[np.zeros(x.shape[1]),np.eye(x.shape[1])]]
    opts=[least_squares(res,v,bounds=(-5,5),max_nfev=65) for v in starts]
    opt=min(opts,key=lambda o:o.cost); d=np.c_[a,design(opt.x)]; return d[n:]@np.linalg.lstsq(d[:n],t,rcond=None)[0]
options=[]
for ib in range(4):
    for gr in [(),(0,),(1,),(2,),(3,),(4,),(0,3),(2,3),(0,1,2,3,4)]:
        for mode, vals in [('ridge',[.03,.2,1.,5.,20.]),('sparse',[1,2,4,7])]:
            for val in vals if gr else ([0] if mode=='ridge' else []):options.append(('regular',(ib,gr,mode,val)))
for dims in ['z','u','uz','w','zw']:
    for fun in ['sin','tanh','square','rational','cube']:options.append(('proj',(0,dims,fun)))
# Add a fixed-coefficient trajectory-shaped prior through the paired sum, without using test targets.
zz,uu=z,u
cache=[]
for a,b in allfold:
    ids=np.r_[a,b]
    tmp=dict(s); tmp['train']={k:s['train'][k][a] for k in ('x','y','traj')}; tmp['test']={'x':x[b]}
    old=run_node({'__input__':tmp,'node_00':{'z_tr':z[a],'z_te':z[b]}})
    cache.append((a,b,ids,old['y_hat'],float(np.mean(abs(old['state_pred']-Y[b,1:])))))
res=[]
for typ,spec in options:
    preds=[]
    for a,b,ids,old,et in cache:
        pred=(_label_model(uu[ids],zz[ids],t[a],spec) if typ=='regular' else nonlinear(uu[ids],zz[ids],t[a],spec))
        preds.append(pred)
    e=np.array([np.mean(abs(pr-t[b])) for pr,(_,b,*_) in zip(preds,cache)])
    res.append((float(np.mean(e[1:])+.4*np.max(e[1:])),e,typ,spec,preds))
res.sort(key=lambda x:x[0]); print('best label hypotheses',[(typ,sp,e.round(4).tolist()) for _,e,typ,sp,_ in res[:12]],flush=True)
old=[q[3] for q in cache]; olde=np.array([np.mean(abs(pr-t[b])) for pr,(_,b,*_) in zip(old,cache)]); print('baseline',olde.round(4),'trajectory', [round(q[-1],5) for q in cache],flush=True)
# Test a small consensus of structurally distinct candidates as an alternative to a single fit.
chosen=[]; families=set()
for row in res:
    _,_,typ,spec,_=row; key=(typ,spec[1],spec[2])
    if key not in families:chosen.append(row);families.add(key)
    if len(chosen)==6:break
strategies=[]
for rows in [[row] for row in res[:25]]+[chosen[:j] for j in (2,3,5,6)]:
    for blend in (0.,.25,.5):
        pr=[(1-blend)*np.mean([row[4][i] for row in rows],axis=0)+blend*old[i] for i in range(len(cache))]; e=np.array([np.mean(abs(pp-t[b])) for pp,(_,b,*_) in zip(pr,cache)])
        score=e[1:].mean()+.4*e[1:].max(); strategies.append((score,e,rows,blend))
strategies.sort(key=lambda k:k[0]); passes=[item for item in strategies if item[1][0]<=.038]; winner=(passes or strategies)[0]
score,e,rows,blend=winner
if e[0] > .038 and e[0] > olde[0] and score >= olde[1:].mean()+.4*olde[1:].max():
    print('retained baseline',flush=True)
else:
    print('chosen',e.round(5),[(row[2],row[3]) for row in rows],blend,flush=True)
    nlsrc=''
    if any(row[2]=='proj' for row in rows):
        src=Path(__file__).read_text().split('def nonlinear(',1)[1].split('\noptions=[]',1)[0]; nlsrc='\nfrom scipy.optimize import least_squares\n\ndef _projected_label('+src+'\n'
    specs=[(row[2],row[3]) for row in rows]; calls=', '.join(('_label_model' if typ=='regular' else '_projected_label')+'(u, z, target, '+repr(spec)+')' for typ,spec in specs)
    text=original.replace("    y = _fit(label, np.asarray(tr['y']).reshape(-1), n, .001)","    target = np.asarray(tr['y']).reshape(-1)\n    y = np.mean(["+calls+"], axis=0)"+("\n    y = (1-"+repr(blend)+")*y + "+repr(blend)+"*_fit(label, target, n, .001)" if blend else ''))
    if not blend:text=text[:text.index("    panel = s['panel_c']")]+text[text.index('    trajectory ='):]
    p.write_text(text+'\n'+helpers+nlsrc)
