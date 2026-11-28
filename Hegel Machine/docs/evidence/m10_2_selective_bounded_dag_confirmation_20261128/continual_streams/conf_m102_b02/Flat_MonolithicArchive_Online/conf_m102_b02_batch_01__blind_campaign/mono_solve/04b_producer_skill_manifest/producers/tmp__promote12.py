exec(open('/tmp/cv_exact.py').read().split('P=[np.arange')[0])
from pathlib import Path
from scipy.special import erf
X=tr['x']; T=tr['traj']; a=m._calibration(s['panel_a']); z=np.linalg.solve(a[1:].T,(X-a[0]).T).T
pb,pc=s['panel_b'],s['panel_c']; b=m._operator(pb['u'],pb['v'],erf); c=m._operator(pc['u'],pc['v1'],np.arcsinh); r,h=c(z),c(b(z)); base=np.column_stack((z,r,h,z*r,z*h,r*h))
P=[np.arange(N)]+[rng.permutation(N) for _ in range(18)]; gen=np.random.default_rng(1102); P.extend(gen.permutation(N) for _ in range(16)); P.extend(np.argsort(d*X[:,j]) for j in range(4) for d in [-1,1]); P.extend(np.argsort(d*z[:,j]) for j in range(4) for d in [-1,1]);
def metrics(designs):
    out=[]
    for p in P:
        i,j=p[:52],p[52:]; V=np.linalg.svd(T[i],full_matrices=False)[2][:2]; pred=sum(m._ridge(D[p],T[i]@V.T,.003,52) for D in designs)/len(designs); e=abs(pred@V-T[j]); out.append([e.mean(),*e.mean(0)])
    out=np.array(out); return out[0],out[1:35].max(0),out[1:35].mean(0),out[35:].max(0),out[35:].mean(0)
a0=metrics([base]); cube=np.column_stack((base,(h**3).sum(1))); logs=np.column_stack((base,np.log1p((z+h)**2))); a1=metrics([cube]); a2=metrics([cube,logs]);
for k,stat in enumerate([a0,a1,a2]): print(k,*stat,sep='\n')
opts=[(k,a) for k,a in [(1,a1),(2,a2)] if a[0][0]<a0[0][0] and a[1][0]<=a0[1][0]+.001 and a[3][0]<.9*a0[3][0] and a[3][-1]<a0[3][-1]]
if opts:
    k=min(opts,key=lambda x:x[1][3][0]+.4*x[1][4][0])[0]; f=Path(m.__file__); text=f.read_text(); old='    state = _ridge(ds, t @ modes.T, .003, n) @ modes'; assert old in text
    new='    amplitudes = t @ modes.T\n    cubic = np.column_stack((ds, (h**3).sum(1)))\n    projected = _ridge(cubic, amplitudes, .003, n)\n'
    if k==2: new+='    logarithmic = np.column_stack((ds, np.log1p((z + h)**2)))\n    projected = .5 * (projected + _ridge(logarithmic, amplitudes, .003, n))\n'
    f.write_text(text.replace(old,new+'    state = projected @ modes')); print('Promoted',k)
else: print('Retained verified baseline')
