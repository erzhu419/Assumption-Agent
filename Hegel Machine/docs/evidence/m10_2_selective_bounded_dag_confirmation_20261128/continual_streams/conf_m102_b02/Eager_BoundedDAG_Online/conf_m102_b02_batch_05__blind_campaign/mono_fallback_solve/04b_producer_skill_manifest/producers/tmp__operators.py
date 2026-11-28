from probe import *
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=5,suppress=True,linewidth=160)
a=S['panel_a'];u=a['u'];_,_,e=cal(S,S['train']['x']);keep=e<2;u=u[keep];v=a['v'][keep]
def check(F,v):
 W=np.linalg.pinv(F); C=W@v;R=F@C-v;h=np.sum(F*W.T,axis=1);return np.std(R),np.mean(abs(R/(1-h[:,None]))),C
for label,F in [('lin',D(u)),('sq',D(np.c_[u,u*u])),('cub',D(np.c_[u,u**3])),('both',D(np.c_[u,u*u,u**3])),('quad',P(u,2)),('q+cub',np.c_[P(u,2),u**3]),('full3',P(u,3)),('roll2',D(np.c_[u,u*np.roll(u,1,axis=1)]))]:
 r,l,c=check(F,v);print('A',label,r,l);print(c if label in ['both','quad','q+cub'] else '')
for key,out in [('panel_a','v'),('panel_b','v'),('panel_c','v2')]:
 p=S[key];a,b=(u,v) if key=='panel_a' else (p['u'],p[out]);print(key)
 for tag,tf in [('sin',np.sin),('tanh',np.tanh),('erf',erf),('asinh',np.arcsinh),('cos',np.cos),('exp',np.exp),('rat',lambda x:x/(1+x*x)),('ratsq',lambda x:x*x/(1+x*x))]:
  for combo in [0,1,2]:
   def ff(q):
    h=tf(q*a);return D(h if combo==0 else np.c_[a,h] if combo==1 else np.c_[a,a*a,h])
   q=min([minimize_scalar(lambda q:check(ff(q),b)[1],bounds=(i,j),method='bounded') for i,j in [(.01,1),(1,2),(2,4),(4,7)]], key=lambda o:o.fun)
   if q.fun<.065: print(tag,combo,round(q.x,5),round(q.fun,7),round(check(ff(q.x),b)[0],7))
 for tag,G in [('sep2',D(np.c_[a,a*a])),('sep3',D(np.c_[a,a*a,a**3])),('onlycross',D(np.c_[a,a*np.roll(a,1,axis=1)])),('cross+cub',D(np.c_[a,a*np.roll(a,1,axis=1),a**3]))]:
  r,l,c=check(G,b);print(tag,r,l);print(c if key=='panel_b' else '')
