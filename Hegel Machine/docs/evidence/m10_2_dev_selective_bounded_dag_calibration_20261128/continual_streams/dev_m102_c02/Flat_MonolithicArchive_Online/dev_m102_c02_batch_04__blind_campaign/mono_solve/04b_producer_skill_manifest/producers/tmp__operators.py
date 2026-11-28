from discover import *
from itertools import permutations

def op(u,v,kind):
 def f(x,p):
  if kind=='cubic':return x+p[0]*x**3
  if kind=='tanh':return np.tanh(np.exp(p[0])*x)
  if kind=='sin':return np.sin(np.exp(p[0])*x)
  if kind=='atan':return np.arctan(np.exp(p[0])*x)
  if kind=='rational':return x/(1+np.exp(p[0])*x*x)
  if kind=='power':return x*np.abs(x)**np.exp(p[0])
  if kind=='sq':return x+p[0]*x*x
  if kind=='blend':return np.c_[x,np.tanh(np.exp(p[0])*x)]
  if kind=='diagsq':return np.c_[x,x*x]
  if kind=='diagcub':return np.c_[x,x**3]
  return x
 def prof(p,x=u):return aff(f(x,p))
 def err(p):
  H=prof(p);return (H@L(H,v,rcond=None)[0]-v).ravel()
 if kind in ('diagsq','diagcub','linear'):best=np.array([0.])
 else:best=min((least_squares(err,[i],bounds=(-8,3),max_nfev=100).x for i in [-2,0,1]),key=lambda p:np.mean(err(p)**2))
 h=prof(best);w=L(h,v,rcond=None)[0];e=err(best).reshape(v.shape);lev=np.sum(h*np.linalg.pinv(h).T,axis=1)
 return np.sqrt(np.mean((e/np.maximum(1e-8,1-lev[:,None]))**2)),best,w,lambda x:prof(best,x)@w

if __name__=='__main__':
 z=calibrate()(s['train']['x']); y=s['train']['y']; T=s['train']['traj'];V=np.c_[y,T]
 ops={}
 for n,k,v in [('B','panel_b','v'),('C','panel_c','v1'),('D','panel_c','v2')]:
  values={a:op(s[k]['u'],s[k][v],a) for a in ['linear','cubic','diagcub','tanh','sin','atan','rational','power','blend','sq','diagsq']}
  for a,(e,p,w,f) in sorted(values.items(),key=lambda kv:kv[1][0]): print(n,a,round(e,6),p)
  a,(e,p,w,f)=min(values.items(),key=lambda kv:kv[1][0]);print(n,'best coefficients',w);ops[n]=f
 def score(H):
  pred=H[52:]@L(H[:52],V[:52],rcond=None)[0];e=np.mean(abs(pred-V[52:]),0);return np.round([e[0],e[1:].mean(),e[-1]],5)
 xs={'Z':z,'Zt':np.tanh(z)}
 for m in range(1,4):
  for path in permutations(ops,m):
   a=z
   for k in path:a=ops[k](a)
   xs[''.join(path)]=a
 for name,a in xs.items():
  c=a[:,:2]*a[:,2:]; r=a*np.roll(a,-1,axis=1)
  print(name,*[(lab,score(h)) for lab,h in [('aff',aff(a)),('sq',np.c_[aff(a),a*a]),('opp',np.c_[aff(a),c.sum(1)]),('quad',mon(a,2)),('cubic',np.c_[aff(a),a**3]),('tanh',aff(np.tanh(a))),('quad_tanh',mon(np.tanh(a),2))]])
