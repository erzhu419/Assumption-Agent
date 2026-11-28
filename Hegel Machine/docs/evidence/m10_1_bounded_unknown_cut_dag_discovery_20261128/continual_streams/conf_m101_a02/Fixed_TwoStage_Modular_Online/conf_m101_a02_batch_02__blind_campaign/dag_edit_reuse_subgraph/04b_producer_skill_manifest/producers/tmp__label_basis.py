import numpy as np, pickle, itertools, runpy
P='/holdout_task/'
st=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; f=runpy.run_path(P+'_frozen_nodes/node_00.py'); enc=f['run_node']({'__input__':st}); y=st['train']['y']; z=enc['z_tr']; a,b=f['_calibration'](st['panel_a']); x=(st['train']['x']-a)/b
print('z range',z.min(0),z.max(0),'test range',enc['z_te'].min(0),enc['z_te'].max(0),'y range',y.min(),y.max()); print('mixer', f['_fit'](f['_affine'](st['panel_b']['u']),st['panel_b']['v']))
rng=np.random.default_rng(19); groups=[np.arange(54,72)]+[rng.permutation(len(y))[:18] for _ in range(16)]; groups += [np.argsort(abs(z[:,j]))[-12:] for j in range(4)];
def fit_eval(F,tr,te,alpha=0):
    A=F[tr]; scale=np.sqrt((A*A).mean(0)); scale[scale<1e-12]=1; A=A/scale
    if alpha:
        pen=np.eye(A.shape[1])*alpha; pen[0,0]=0; w=np.linalg.solve(A.T@A+pen,A.T@y[tr]);
    else: w=np.linalg.lstsq(A,y[tr],rcond=None)[0]
    return np.mean(abs(F[te]/scale@w-y[te]))
def score(F,alpha=0):
    e=[fit_eval(F,np.setdiff1d(np.arange(len(y)),g),g,alpha) for g in groups]; return np.round([e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],6)
def poly(p,d):
    ii=[i for k in range(1,d+1) for i in itertools.combinations_with_replacement(range(4),k)]; return np.column_stack([np.ones(len(p))]+[p[:,i].prod(1) for i in ii]),['1']+[''.join(map(str,i)) for i in ii]
def greedy(F,tr,k):
    A=F[tr]; resid=y[tr].copy(); sel=[0];
    while len(sel)<k:
        q=np.linalg.qr(A[:,sel])[0]; G=A-q@(q.T@A); norm=np.linalg.norm(G,axis=0); obj=abs(G.T@resid)/np.maximum(norm,1e-9); obj[sel]=-1; sel.append(np.argmax(obj)); resid=y[tr]-A[:,sel]@np.linalg.lstsq(A[:,sel],y[tr],rcond=None)[0]
    return sel
for label,p in [('z',z),('x',x)]:
    for d in [2,3,4,5]:
        F,terms=poly(p,d); print(label,d,[(r,score(F,r).tolist()) for r in (0,.001,.01,.1,1)])
        if d in (3,5):
            for k in (7,10,14,20):
                sel=greedy(F,np.arange(len(y)),k); e=[fit_eval(F[:,greedy(F,np.setdiff1d(np.arange(len(y)),g),k)],np.setdiff1d(np.arange(len(y)),g),g) for g in groups]; print('sparse',k,[terms[j] for j in sel], np.round([e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],6))
    Q,_=poly(p,2)
    for func in (np.sin,np.tanh,np.arctan):
        for freq in (.7,1,1.5,2,2.5,3):
            P=p*freq; S=func(P); pairs=list(itertools.combinations(range(4),2)); I=np.column_stack([func(P[:,i]+s*P[:,j]) for i,j in pairs for s in (1,-1)]); T=np.column_stack([S[:,i]*p[:,j] for i,j in itertools.permutations(range(4),2)]); print(label,func.__name__,freq,score(np.column_stack([Q,S])),score(np.column_stack([Q,S,I]),.03),score(np.column_stack([Q,S,T]),.03))
