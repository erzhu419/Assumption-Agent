import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['panel_c'];x=np.asarray(p['u']);v=p['v2'];n=len(x);a,b=np.split(np.random.default_rng(53).permutation(n),[18]);y=np.roll(x,-1,axis=1);o=np.ones(n);base=np.c_[o,x]
def ls(d,v):return np.linalg.lstsq(d,v,rcond=1e-11)[0]
def rms(r):return np.mean(r*r)**.5
funcs={'cos':np.cos,'sin':np.sin,'rat':lambda t:1/(1+t*t),'exp':lambda t:np.exp(-t*t),'soft':lambda t:t/(1+abs(t)),'tanh':np.tanh}
forms={}
for k,f in funcs.items():
 for mode in range(7):
  def d(p,x,y,k=k,f=f,mode=mode):
   a,b,c=p
   if mode==0:return (x+b)*(y+c)*f(a*(x-y))
   if mode==1:return (x+b)*(y+c)*f(a*(x+y))
   if mode==2:return (x+b)*(y+c)*f(a*x*y)
   if mode==3:return f(a*x+b)*f(a*y+c)
   if mode==4:return f(a*(x+b*y)+c)
   if mode==5:return (x+b)*f(a*y+c)
   if mode==6:return (x*y+c)*f(a*(x+b*y))
  forms[k,mode]=d
for mode in range(10):
 def d(p,x,y,mode=mode):
  a,b,c=p
  if mode==0:return (x+b)*(y+c)/(1+a*x*x+a*y*y)
  if mode==1:return (x+b)*(y+c)/(1+a*x*x)
  if mode==2:return (x+b)*(y+c)/(1+a*y*y)
  if mode==3:return x*y/(1+a*x*x+b*b*y*y+c*c*(x-y)**2)
  if mode==4:return (x*y+b*(x+y)+c)/(1+a*(x-y)**2)
  if mode==5:return (x*y+b*(x-y)+c)/(1+a*(x+y)**2)
  if mode==6:return (x*y+b*(x+y)+c)/(1+a*(x+y)**2)
  if mode==7:return (x*y+b*(x-y)+c)/(1+a*(x-y)**2)
  if mode==8:return (x*y+b*x*x+c*y*y)/(1+a*(x-y)**2)
  if mode==9:return (x*y+b*x*x+c*y*y)/(1+a*(x+y)**2)
 forms['rat2',mode]=d
out=[]
for ori in [1,-1,2]:
 y=np.roll(x,-ori,axis=1)
 for name,d in forms.items():
  def design(p):return np.c_[base,d(p,x,y)]
  def obj(p):
   m=design(p)[a];return (m@ls(m,v[a])-v[a]).ravel()
  fits=[least_squares(obj,t,bounds=([.015,-5,-5],[9,5,5]),max_nfev=105) for t in [(.45,.3,-.2),(1.25,-.6,-.6),(2.15,1,1),(.6,-1.5,1.5)]]
  q=min(fits,key=lambda f:f.cost).x;m=design(q);out.append((rms(m[b]@ls(m[a],v[a])-v[b]),rms(obj(q)),ori,name,q,ls(m[a],v[a])[5:].round(3)))
for r in sorted(out,key=lambda r:r[0])[:14]:print(r[:5]);print(r[5] if r[0]<.02 else '')
for r in sorted(out,key=lambda r:r[1])[:6]:print('in',r[:5])
print('panels',[(k,[(j,np.shape(w)) for j,w in z.items()]) for k,z in s.items() if k.startswith('panel')])
