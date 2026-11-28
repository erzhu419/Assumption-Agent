exec(open('/tmp/probe.py').read().split('results={}')[0])
from itertools import combinations
p=s['panel_c']; a,b=p['v1'],p['v2']; n=len(z)
def design(t, v1, v2):
 f,g=v1(t),v2(t); h=_bias(np.column_stack([f,g])); return np.column_stack([h,np.sum(f*g,axis=1),f[:,0]*g[:,1]])
A=lambda t: t/(1+1.25*abs(t)); P=lambda t:t*np.roll(t,1,axis=1)
ma=_bias(np.column_stack([A(u),u/(1+.956**2*u*u)])); ca=_ls(ma,a)
mb=_bias(np.column_stack([np.arctan(1.20563*u),P(u)])); cb=_ls(mb,b)
print('coeff v1',ca.round(5));print('coeff v2',cb.round(5));print('v2 shared relative',np.linalg.solve(cb[1:5].T,cb[5:].T).T.round(5))
f=lambda t:_bias(np.column_stack([A(t),t/(1+.956**2*t*t)]))@ca
g=lambda t:_bias(np.column_stack([np.arctan(1.20563*t),P(t)]))@cb
F=design(z,f,g); w1,w2=f(z),g(z)
models={'archive':F,'linear':F[:,:9],'diag':np.column_stack([F[:,:9],w1*w2]),'cross':np.column_stack([F,(w1[:,:,None]*w2[:,None,:]).reshape(n,-1)]),'poly':np.column_stack([F]+[w1[:,i]*w1[:,j] for i in range(4) for j in range(i,4)]+[w2[:,i]*w2[:,j] for i in range(4) for j in range(i,4)]+[(w1[:,:,None]*w2[:,None,:]).reshape(n,-1)]),'latent':np.column_stack([F,P(z), z, z*z, z*z*z])}
rng=np.random.default_rng(11); splits=[np.arange(52,72)]+[rng.permutation(n)[:20] for j in range(12)]
for name,F in models.items():
 E=[]
 for te in splits:
  tr=np.setdiff1d(np.arange(n),te); err=np.abs(F[te]@_ls(F[tr],Y[tr])-Y[te]); E.append([err[:,0].mean(),err[:,1:].mean()])
 print(name,'official',np.round(E[0],6),'worst',np.max(E,axis=0).round(6),'resid',np.mean(abs(F@_ls(F,Y)-Y),axis=0).round(6))
print('archive fit coefficients',_ls(models['archive'],Y).round(4))
