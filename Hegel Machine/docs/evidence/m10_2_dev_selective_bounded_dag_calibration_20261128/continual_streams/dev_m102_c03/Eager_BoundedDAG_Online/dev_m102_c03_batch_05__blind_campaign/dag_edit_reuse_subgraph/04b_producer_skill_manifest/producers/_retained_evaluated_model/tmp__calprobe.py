exec(open('/tmp/readout_probe.py').read().split("for kind in ['diag'")[0]); u,v=(s['panel_a'][k] for k in ('u','v')); h=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3; u,v=u[keep],v[keep]; a=u;b=np.roll(u,-1,axis=1); c=np.roll(u,-2,axis=1); d=np.roll(u,-3,axis=1)
def obj(fs,ret=False):
 L=[];B=[]
 for i in range(4):
  h=np.c_[np.ones(len(a)),a[:,i],b[:,i],(a*b)[:,i],*[f[:,i] for f in fs]]; q,r=np.linalg.qr(h); e=(v[:,i]-q@(q.T@v[:,i])); L.append(e/(1-(q*q).sum(1))); B.append(np.linalg.lstsq(h,v[:,i],rcond=None)[0]);
 return np.sqrt(np.mean(np.array(L)**2)) if not ret else np.array(B).T
from scipy.optimize import minimize_scalar
print('base',obj([]));out=[]
for name,g in [('a',a),('b',b),('c',c),('d',d),('a+b',a+b),('a-b',a-b),('c+d',c+d),('c-d',c-d),('ab',a*b),('ac',a*c),('ad',a*d),('a+c',a+c),('a-c',a-c),('a+d',a+d),('a-d',a-d),('b+c',b+c),('b+d',b+d)]:
 for fun in (np.sin,np.cos,np.tanh,np.arctan):
  opt=minimize_scalar(lambda w:obj([fun(w*g)]),bounds=(.05,6),method='bounded');out.append((opt.fun,name,fun.__name__,opt.x,obj([fun(opt.x*g)],True)[-1]))
print(*sorted(out,key=lambda a:a[0])[:15],sep='\n');print('poly',[(k,obj([g])) for k,g in [('a3',a**3),('b3',b**3),('c3',c**3),('d3',d**3),('a2b',a*a*b),('ab2',a*b*b),('abc',a*b*c),('abd',a*b*d),('c',c),('d',d),('a2',a*a),('b2',b*b)]]);print('full quadratic',obj([a*a,b*b,c,d,c*c,d*d,a*c,a*d,b*c,b*d,c*d]));print('residual',v-np.column_stack([np.c_[np.ones(len(a)),a[:,i],b[:,i],a[:,i]*b[:,i]]@obj([],True)[:,i] for i in range(4)]))
