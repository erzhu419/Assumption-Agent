"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "6ee427ea7efa40d4df9d280699dd9011712b084bec3b7736c1ae4fc2104f1ef9"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\nfrom itertools import combinations_with_replacement\\n\\n_I = np.arange(4)\\n_J = (_I + 1) % 4\\n\\ndef _inverse(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    def basis(a, j):\\n        b = (j + 1) % 4\\n        return np.column_stack((np.ones(len(a)), a[:,j], a[:,b], a[:,j]*a[:,b], a[:,j]**3))\\n    matrices = [basis(u,j) for j in _I]\\n    weights = np.ones(len(u))\\n    for _ in range(20):\\n        coeff = np.array([np.linalg.lstsq(m*weights[:,None], v[:,j]*weights, rcond=None)[0] for j,m in enumerate(matrices)])\\n        err = np.linalg.norm(v-np.column_stack([m@c for m,c in zip(matrices,coeff)]),axis=1)\\n        weights = 1/np.hypot(1,err/max(1e-4,1.8*np.median(err)))\\n    k = np.argsort(err)[:-3]\\n    coeff = np.array([np.linalg.lstsq(m[k],v[k,j],rcond=None)[0] for j,m in enumerate(matrices)])\\n    linear = np.diag(coeff[:,1]); linear[_I,_J] = coeff[:,2]\\n    def residual(a, row):\\n        return coeff[:,0]+linear@a+coeff[:,3]*a*a[_J]+coeff[:,4]*a**3-row\\n    return np.array([least_squares(residual,np.linalg.solve(linear,row-coeff[:,0]),args=(row,)).x for row in x])\\n\\ndef _panel_b(panel, x):\\n    u,v=panel[\'u\'], panel[\'v\']\\n    def design(u, a):\\n        return np.c_[np.ones(len(u)),u,np.sign(u)*np.log1p(a[0]*abs(u)**a[1])]\\n    def residual(a):\\n        m=design(u,a); p=np.linalg.pinv(m)\\n        return ((m@(p@v)-v)/(1-(m*p.T).sum(1))[:,None]).ravel()\\n    a=least_squares(residual,[1.3,3.2],bounds=([.005,1.5],[7,6])).x\\n    return design(x,a)@np.linalg.lstsq(design(u,a),v,rcond=None)[0]\\n\\ndef _polys(x, degree):\\n    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for d in range(1,degree+1) for k in combinations_with_replacement(range(4),d)])\\n\\n\\nfrom scipy.linalg import solve\\n\\ndef _calibrate_b(panel,x,key,extra,theta):\\n    if key==\\"baseline\\": return _panel_b(panel,x)\\n    u,v=panel[\\"u\\"],panel[\\"v\\"]\\n    i,j=np.triu_indices(4,1)\\n    def design(x,a):\\n        fs={\\n            \'csat\':lambda x,a:np.tanh(a[0]*x)**a[1] if False else np.sign(x)*np.tanh(a[0]*abs(x))**a[1],\\n        }\\n        g=fs[key](x,a)\\n        extras={\\"lin\\":[x],\\"none\\":[],\\"quad\\":[x,x*x],\\"cyc\\":[x,x*np.roll(x,-1,axis=1)],\\"cross\\":[x,x[:,i]*x[:,j]],\\"cube\\":[x,x**3],\\"mix\\":[x,x*x,x*np.roll(x,-1,axis=1)]}\\n        return np.column_stack([np.ones(len(x)),g]+extras[extra])\\n    def residual(a):\\n        F=design(u,a); H=np.linalg.pinv(F,rcond=1e-10)\\n        return ((v-F@(H@v))/np.maximum(.05,1-(F*H.T).sum(1))[:,None]).ravel()\\n    lo,hi=([.001,.03],[8,6]) if key not in (\\"poly\\",\\"ratmix\\") else ([-.35,-3],[5,3])\\n    if key==\\"ratmix\\": lo=[.001,-3]\\n    a=least_squares(residual,theta,bounds=(lo,hi),max_nfev=50).x\\n    return design(x,a)@np.linalg.lstsq(design(u,a),v,rcond=1e-10)[0]\\ndef pf(b,typ,a,w,c1):\\n i,j=np.triu_indices(4,1)\\n P=_polys(b,2); b0,b1=b[:,i],b[:,j]; qp=b0*b1; plus=b0+b1; minus=b0-b1\\n if typ==\'poly\':return _polys(b,int(a))\\n if typ==\'bounded\':return np.c_[b,np.tanh(a*b),np.tanh(a*P[:,5:])]\\n if typ==\'satcubic\':return np.c_[P,np.tanh(a*_polys(b,3)[:,15:])/a]\\n if typ==\'pairshape\':return np.c_[P,qp/(1+a*plus**2),qp/(1+a*minus**2),np.sin(a*qp),np.cos(a*qp)]\\n if typ==\'pairseparable\': return np.c_[P,qp/(1+a*(b0*b0+b1*b1)),qp/(1+a*qp**2),qp/(1+a*plus**2),qp/(1+a*minus**2)]\\n if typ==\'quotient\':return np.c_[P,(P[:,:,None]/(1+a*plus[:,None,:]**2)).reshape(len(P),-1)]\\n if typ==\'cyclequot\':return np.c_[P,(P[:,:,None]/(1+a*(b+np.roll(b,-1,axis=1))[:,None,:]**2)).reshape(len(P),-1)]\\n if typ==\'mixed\':return np.c_[P,w,c1,(b[:,:,None]*c1[:,None,:]).reshape(len(P),-1),qp/(1+a*plus**2),qp/(1+a*minus**2)]\\n if typ==\'axis\':return np.c_[P,(b[:,:,None]*np.tanh(a*b)[:,None,:]).reshape(len(P),-1),qp/(1+a*plus**2),qp/(1+a*minus**2)]\\ndef fold_profile(z,t):\\n def profile(k):return -np.expm1(-k*t)\\n def err(k):\\n  q=profile(k); return np.linalg.norm(z-np.outer(z@q/(q@q),q))\\n return profile(minimize_scalar(err,bounds=(.1,2),method=\'bounded\').x)\\n\\n\\ndef _new_b(panel, x, mode):\\n    u,v=panel[\'u\'],panel[\'v\']\\n    def phi(x,t):\\n        a,b=t\\n        if mode==0: return np.tanh(a*x**3)\\n        if mode==1: return np.tanh(a*x**3+b*x)\\n        if mode==2: return np.sin(a*x**3+b*x)\\n        if mode==3: return np.arcsinh(a*x**3+b*x)\\n        if mode==4: return np.tanh(a*x)+b*np.sin(1.7*x)\\n        if mode==5: return np.tanh(a*x**3)+b*np.tanh(1.7*x)\\n        if mode==6: return np.sin(a*x)+b*np.sin(1.7*x)\\n        if mode==7: return np.tanh(a*x)+b*np.tanh(1.7*x)\\n        if mode==8: return np.tanh(a*x)+b*x**3\\n        if mode==9: return np.tanh(a*x**3/(1+abs(b)*x*x))\\n        return np.tanh(a*x)+b*x**3/(1+x*x)\\n    def f(x,t): return np.c_[np.ones(len(x)),x,phi(x,t)]\\n    def r(t):\\n        F=f(u,t); H=np.linalg.pinv(F,rcond=1e-9)\\n        return ((F@(H@v)-v)/np.maximum(.04,1-(F*H.T).sum(1))[:,None]).ravel()\\n    fit=min((least_squares(r,t,bounds=([.05,-4],[8,4]),max_nfev=65) for t in ((.8,.15),(1.7,-.25),(3,.6))),key=lambda o:o.cost)\\n    th=fit.x; F=f(u,th)\\n    return f(x,th)@np.linalg.lstsq(F,v,rcond=1e-9)[0],np.mean(r(th)**2)**.5\\n\\ndef _scalar_features(b,kind,k):\\n    P=_polys(b,3); C=P[:,15:]; Q=P[:,5:15]\\n    if kind==0: return np.c_[P[:,:15],np.tanh(k*C)/k]\\n    if kind==1: return np.c_[P[:,:15],np.arcsinh(k*C)/k]\\n    if kind==2: return np.c_[P[:,:15],C/(1+k*C*C)]\\n    if kind==3: return np.c_[P[:,:5],np.tanh(k*Q)/k,np.tanh(k*C)/k]\\n    if kind==4: return np.c_[P[:,:15],np.tanh(k*C)/k,np.sin(k*Q)/k]\\n    if kind==5: return np.c_[P[:,:15],np.tanh(k*C)/k,np.tanh(k*_polys(b,4)[:,35:])/k]\\n    if kind==6: return np.c_[P[:,:15],np.tanh(k*C)/k,C]\\n    if kind==7: return np.c_[P[:,:5],Q,np.tanh(k*Q)/k,np.tanh(k*C)/k]\\n    if kind==8: return np.c_[P[:,:15],np.tanh(k*C)/k,_polys(np.tanh(k*b)/k,2)[:,1:]]\\n    i,j=np.triu_indices(4,1); S=(b[:,i]+b[:,j])**2; D=(b[:,i]-b[:,j])**2\\n    return np.c_[P[:,:15],np.tanh(k*C)/k,(Q[:,:,None]/(1+k*np.c_[S,D][:,None,:])).reshape(len(b),-1)]\\n\\ndef _scalar_fit(F,b,y,n,lam,scaling,link):\\n    A=F[:n]; scale=np.maximum(.25,A.std(0)) if scaling else 1.; G=(F-A.mean(0))/scale; T=G[:n]; mean=y.mean()\\n    if link in (1,2,-1,-2):\\n        c=max(abs(y))*({1:1.05,2:1.3,-1:1.15,-2:1.6}[link]); v=c*np.arctanh(y/c) if link>0 else c*np.tanh(y/c)\\n        mean=v.mean(); z=G[n:]@T.T@solve(T@T.T+lam*np.eye(n),v-mean,assume_a=\'pos\')+mean\\n        return c*np.tanh(z/c) if link>0 else c*np.arctanh(np.clip(z/c,-.97,.97))\\n    if link==3:\\n        q=np.c_[b,b[:,np.arange(4)]**2,_polys(b,2)[:,5:]]; q=q/np.maximum(.3,q[:n].std(0)); X=np.c_[np.ones(n),T,-y[:,None]*q[:n]]\\n        reg=np.r_[1e-8,np.full(T.shape[1],lam),np.full(q.shape[1],lam*12+.02)]; X1=X/np.sqrt(reg); c=(X1.T@solve(X1@X1.T+np.eye(n),y,assume_a=\'pos\'))/np.sqrt(reg)\\n        return (c[0]+G[n:]@c[1:1+T.shape[1]])/np.maximum(.4,1+q[n:]@c[1+T.shape[1]:])\\n    return G[n:]@T.T@solve(T@T.T+lam*np.eye(n),y-mean,assume_a=\'pos\')+mean\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s=inputs.get(\\"__input__\\",inputs); tr=s[\\"train\\"]; n=len(tr[\\"x\\"])\\n    w=_inverse(s[\\"panel_a\\"],np.r_[tr[\\"x\\"],s[\\"test\\"][\\"x\\"]]); t=s[\\"eval_times\\"]\\n    q=fold_profile(tr[\\"traj\\"],t)\\n    Y=np.c_[tr[\\"y\\"],tr[\\"traj\\"]@q/(q@q)]\\n    c1=None\\n    preds=[]\\n    for col,(key,extra,theta,typ,a,lam,scaling) in enumerate([\\n        (\'csat\', \'lin\', [1.7019903686123374, 1.0805054637640883], \'satcubic\', 3.0, 0.02, True),\\n        (\'csat\', \'cube\', [1.7149165936508188, 5.999999999999978], \'satcubic\', 3.0, 0.08, True),\\n    ]):\\n        b=_calibrate_b(s[\\"panel_b\\"],w,key,extra,theta)\\n        F=pf(b,typ,a,w,c1); A=F[:n]; center=A.mean(0); scale=np.maximum(.25,A.std(0)) if scaling else 1.\\n        A=(A-center)/scale; G=(F[n:]-center)/scale\\n        v=Y[:,col]; m=v.mean(); preds.append(G@A.T@solve(A@A.T+lam*np.eye(n),v-m,assume_a=\\"pos\\")+m)\\n    b_scalar=_new_b(s[\\"panel_b\\"],w,7)[0]\\n    Fs=_scalar_features(b_scalar,2,0.6)\\n    y=_scalar_fit(Fs,b_scalar,tr[\\"y\\"],n,0.045,True,2); z=np.outer(preds[1],q)\\n    return {\\"y_hat\\":y,\\"state_pred\\":z,\\"diagnostics_vec\\":np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}\\n"}'

def _canon_digest(obj) -> str:
    h = hashlib.sha256()
    def _feed(v):
        if v is None: h.update(b'N;')
        elif isinstance(v, bool): h.update(b'B1;' if v else b'B0;')
        elif isinstance(v, (int, np.integer)): h.update(f'I:{int(v)};'.encode())
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv): h.update(b'F:NaN;')
            elif math.isinf(fv): h.update(b'F:Inf;')
            else: h.update(b'F:' + np.asarray([0.0 if fv == 0.0 else fv], dtype='<f8').tobytes() + b';')
        elif isinstance(v, str): h.update(f'S:{len(v)}:{v};'.encode())
        elif isinstance(v, np.ndarray):
            h.update(f'A:{v.dtype.kind}:{v.shape}:'.encode())
            if v.dtype.kind == 'f':
                a = np.asarray(v, dtype=np.float64)
                nm = np.isnan(a)
                cl = np.where(nm, 0.0, a)
                cl = np.where(cl == 0.0, 0.0, cl).astype('<f8')
                h.update(cl.tobytes('C'))
            else:
                h.update(np.asarray(v, dtype='<i8').tobytes('C'))
            h.update(b';')
        elif isinstance(v, dict):
            h.update(f'D:{len(v)}:'.encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b';')
        elif isinstance(v, (list, tuple)):
            h.update(f'L:{len(v)}:'.encode())
            for it in v: _feed(it)
            h.update(b';')
        else:
            h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _load_node_fn(nid: str, src: str):
    ns = {'__name__': f'_hegel_node_{nid}'}
    exec(compile(src, f'<dag_node_{nid}>', 'exec'), ns)
    fn = ns.get('run_node') or ns.get('solve_batch')
    if not callable(fn):
        raise RuntimeError(f'DAG node {nid} missing callable run_node/solve_batch')
    return fn

_COMPILED_FNS = {nid: _load_node_fn(nid, src) for nid, src in json.loads(_NODE_SOURCES).items()}
_SPEC_OBJ = json.loads(_DAG_SPEC)

def solve_batch(study: dict) -> dict:
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
    trace_fp = os.environ.get('HEGEL_M101_DAG_TRACE_PATH')
    if trace_fp:
        trace_obj = {
            'dag_id': _SPEC_OBJ['dag_id'],
            'active_node_count': len(_SPEC_OBJ['topo_order']),
            'depth': _SPEC_OBJ['depth'],
            'max_fanout': _SPEC_OBJ['max_fanout'],
            'shared_fanout_nodes': _SPEC_OBJ['shared_fanout_nodes'],
            'topo_order': _SPEC_OBJ['topo_order'],
            'exec_counts': exec_counts,
            'all_exec_count_eq_1': all(v == 1 for v in exec_counts.values()),
            'non_identity_checks': non_identity_checks,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
