import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(x)
one=lambda x:np.c_[np.ones(len(x)),x]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
p=s['panel_a']; U=one(p['u']); V=p['v']; ids=np.arange(len(U));
for i in range(7):
 M=ls(U[ids],V[ids]); err=np.linalg.norm(U@M-V,axis=1); ids=np.argsort(err)[:45]; print('A clipped',np.sort(err)[41:])
print('A',M,'gram',M[1:]@M[1:].T)
p=s['panel_b']; B=ls(one(p['u']),p['v']); p=s['panel_c']; u=p['u']; branches=[]
for v,fun,start in [(p['v1'],lambda w:w/(1+abs(w)),.84),(p['v2'],lambda w:1/(1+w*w),.402)]:
 def fit(u,v,fun,start):
  def feat(z,k):return one(np.c_[z,fun(k*z)])
  def resid(k):D=feat(u,k);return (D@ls(D,v)-v).ravel()
  k=least_squares(resid,[start],gtol=1e-14).x[0]; C=ls(feat(u,k),v);print('C',k,np.std(resid(k)),C);return lambda z:feat(z,k)@C
 branches.append(fit(u,v,fun,start))
H=lambda p:-np.expm1(-t[:,None]*p); h=H(least_squares(lambda p:(H(p)@ls(H(p),T[:52].T)-T[:52].T).ravel(),[1.8,.4]).x); A=ls(h,T.T).T
paths={'x':x}; ops={'A':lambda z:one(z)@M,'a':lambda z:(z-M[0])@np.linalg.inv(M[1:]),'B':lambda z:one(z)@B,'b':lambda z:(z-B[0])@np.linalg.inv(B[1:])}
for k,f in ops.items():paths[k]=f(x)
for k,l in itertools.permutations(ops,2):
 if k.lower()!=l.lower():paths[k+l]=ops[l](paths[k])
for k,f in [('Ad',lambda z: z*np.diag(M[1:])+M[0]),('ad',lambda z:(z-M[0])/np.diag(M[1:]))]:paths[k]=f(x)

def score(D):
 P=D@ls(D[:52],np.c_[y,A][:52]);e=P-np.c_[y,A]; return np.r_[np.mean(abs(e[:52,0])),np.mean(abs(e[52:,0])),np.mean(abs(e[52:,1:]@h.T))]
if __name__=='__main__':
 for k,z in paths.items():
  c,d=[f(z) for f in branches]; m=c*d; cross=(c[:,:,None]*d[:,None,:]).reshape(n,-1); norm=lambda z:np.sum(z*z,axis=1)
  for l,D in [('lin',np.c_[c,d]),('dot',np.c_[c,d,m.sum(1)]),('diag',np.c_[c,d,m]),('diag+',np.c_[c,d,m,norm(c),norm(d),norm(z)]),('cross',np.c_[c,d,cross]),('quad',np.c_[c,d,cross,*[w[:,i]*w[:,j] for w in (c,d) for i in range(4) for j in range(i,4)]])]:print(k,l,np.round(score(one(D)),5))
