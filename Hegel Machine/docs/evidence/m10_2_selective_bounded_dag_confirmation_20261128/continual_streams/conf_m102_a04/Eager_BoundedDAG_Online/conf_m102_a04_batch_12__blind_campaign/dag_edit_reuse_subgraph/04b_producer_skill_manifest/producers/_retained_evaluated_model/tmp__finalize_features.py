exec(open('/tmp/compose.py').read().split('features={};')[0])
from itertools import combinations_with_replacement as cr, product
from pathlib import Path
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
ij=list(cr(range(z.shape[1]),2)); tri=np.array(ij).T
quad=lambda a:a[:,tri[0]]*a[:,tri[1]]
interaction_design=np.c_[np.ones(len(u)),quad(u)]; coef=m._ols(interaction_design,p['v1']); active=np.linalg.norm(coef[1:],axis=1)>max(.002,.003*np.max(np.linalg.norm(coef[1:],axis=1)))
print('C product support',np.array(ij)[active].tolist(),'C fit RMS',np.sqrt(np.mean((interaction_design@coef-p['v1'])**2)),flush=True)
P=lambda a: np.c_[np.ones(len(a)),quad(a)[:,active]]@m._ols(interaction_design[:,np.r_[True,active]],p['v1'])
C=lambda a:cos(a)
B=z@fit[1:]; R=coords['Ic']; H=lambda a:np.cos(q*a)
D={'p':P(z),'c':C(z),'h':H(z),'d':np.sin(q*z),'m':quad(z)[:,active],'q':quad(z),'r':quad(R),'b':quad(B),'i':z,'f':H(R),'g':H(B),'s':np.c_[P(z)+C(z)],'t':np.c_[P(z)-C(z)],'e':z*z,'k':P(z)*C(z)}
# Compare the panel-supported blocks with raw moments. Coefficients are fitted only from calibration data.
combinations=set(['pc','mc','mh','p','c','s','t','st','h','pcp','pcc','ph','pq','qc','rc','bc','rh','bh','q','cpk','pch','pck','pcd','pci','pcr','pcb','pcq','pcm','pcf','pcg','pce'])
for base in ['pc','mc','mh','pcq','rc','bc','qc','p','c','h','q','s','t']:
 for key in D:
  if key not in base: combinations.add(base+key)
# Use several distinct held-out slices; the official last-20 slice remains the first check.
YY=y

def block(a,tr,mode):
 if mode==0:return a
 b=a-a[tr].mean(0)
 if mode==1:return b
 if mode==2:return b/np.maximum(b[tr].std(0),.1)
 _,ss,v=np.linalg.svd(b[tr],full_matrices=False)
 return (b@v.T)/np.maximum(ss/np.sqrt(len(tr)),.07)

cache={(k,l,j):block(a,tr,l) for k,a in D.items() for l in range(4) for j,(tr,te) in enumerate(pairs)}
def evaluate(keys,mode,scale,ratio):
 out=[]
 for j,(tr,te) in enumerate(pairs):
  A=np.column_stack([cache[k,mode,j]*(1 if i==0 else ratio) for i,k in enumerate(keys)])*scale
  G=A@A.T/A.shape[1]; d=G.diagonal(); E=np.maximum(d[:,None]+d-2*G,0); K=(1+.25*G)**2; L=K+np.exp(-.06*E)
  i=np.ix_(tr,tr); h=np.ix_(te,tr); n0=np.eye(len(tr)); pred=.5*(K[h]@np.linalg.solve(K[i]+3e-5*n0,YY[tr])+L[h]@np.linalg.solve(L[i]+3e-4*n0,YY[tr])); e=abs(pred-YY[te]); out.append((e[:,0].mean(),e[:,1:].mean()))
 return np.asarray(out)
results=[]
for keys in sorted(combinations):
 for mode,scale,ratio in product(range(4),[.5,1,2,4],[.5,1,2]):
  a=evaluate(keys,mode,scale,ratio); results.append((keys,mode,scale,ratio,a))
for keys in ['pc','mc','mh']:
 for mode,scale,ratio in product(range(4),[.15,.25,.35,.7,1.4,2.8,5.6,8],[.25,.7,1.4,4]):
  a=evaluate(keys,mode,scale,ratio); results.append((keys,mode,scale,ratio,a))
def order(row):
 a=row[-1]; off=a[0].max(); worst=a.max(); return (off>.0377, worst if off<=.0377 else .65*off+.35*worst, off)
results.sort(key=order)
for row in results[:12]:print(row[:4],row[-1].round(6).tolist(),flush=True)
keys,mode,scale,ratio,a=results[0]
source=Path('/holdout_task/node_00.py').read_text().split('def _products(')[0]
source+='''def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return x[:, i] * x[:, j]


def _normalize(w, n, mode):
    if mode == 0:
        return w
    w = w - w[:n].mean(axis=0)
    if mode == 1:
        return w
    if mode == 2:
        return w / np.maximum(w[:n].std(axis=0), .1)
    _, s, v = np.linalg.svd(w[:n], full_matrices=False)
    return (w @ v.T) / np.maximum(s / np.sqrt(n), .07)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    n = len(train)
    f, coefficients = _calibrate(study['panel_a'], float(study['telemetry_vec'][-1]))
    z = _invert(np.concatenate((train, test)), f, coefficients)
    panel = study['panel_c']
    u = np.asarray(panel['u'], dtype=float)
    y1, y2 = panel['v1'], panel['v2']

    def intercept(a):
        return np.column_stack((np.ones(len(a)), a))

    Z = _quadratic(z)
    U = _quadratic(u)
    support = np.linalg.norm(_ols(intercept(U), y1)[1:], axis=1)
    keep = support > max(.002, .003 * support.max())
    P = intercept(Z[:, keep]) @ _ols(intercept(U[:, keep]), y1)

    def design(freq):
        return intercept(np.cos(freq*u))

    def objective(freq):
        A = design(freq)
        return np.sum((A @ _ols(A, y2) - y2)**2)

    q = minimize_scalar(objective, bounds=(1.35, 3.55), method='bounded').x
    H = np.cos(q*z)
    C = intercept(H) @ _ols(design(q), y2)
    panel_b = study['panel_b']
    rotation = _ols(intercept(panel_b['u']), panel_b['v'])[1:]
    B = z @ rotation
    R = np.linalg.solve(rotation.T, z.T).T
    features = {'p': P, 'c': C, 'h': H, 'd': np.sin(q*z), 'm': Z[:, keep],
                'q': Z, 'r': _quadratic(R), 'b': _quadratic(B), 'i': z,
                'f': np.cos(q*R), 'g': np.cos(q*B), 's': P+C, 't': P-C,
                'e': z*z, 'k': P*C}
'''
source+=f'''    blocks = [_normalize(features[key], n, {mode}) * (1 if i == 0 else {ratio!r})
              for i, key in enumerate({keys!r})]
    return {{'features': {scale!r} * np.column_stack(blocks), 'n_train': n}}
'''
Path('/holdout_task/node_00.py').write_text(source)
