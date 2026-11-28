exec(open('/tmp/probe.py').read().split("p=s['panel_a'];")[0])
from itertools import combinations
z=_coords(s['panel_a'],np.r_[x,s['test']['x']]); p=s['panel_c']; u=p['u']; V=p['v1'];
def cyc(w):return w*np.roll(w,-1,axis=1)
def quad(w):return np.column_stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)])
def allprod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
Fams={'tanh':lambda w,q:np.tanh(q*w),'sin':lambda w,q:np.sin(q*w),'soft':lambda w,q:w/(1+q*abs(w)),'rat':lambda w,q:w/(1+q*w*w),'atan':lambda w,q:np.arctan(q*w)}
extras={'none':lambda w:w[:,:0],'lin':lambda w:w,'cyc':cyc,'quad':quad,'lincyc':lambda w:np.c_[w,cyc(w)],'linquad':lambda w:np.c_[w,quad(w)],'cube':lambda w:w**3,'lincube':lambda w:np.c_[w,w**3],'rollcube':lambda w:w*w*np.roll(w,-1,axis=1),'linrollcube':lambda w:np.c_[w,w*w*np.roll(w,-1,axis=1)]}
res=[]
for name,f in Fams.items():
 for en,e in extras.items():
  def ff(w,q):return one(np.c_[f(w,q),e(w)])
  def obj(q):
   A=ff(u,q);return np.std(V-A@ls(A,V))
  g=np.geomspace(.03,6,70);k=np.argmin([obj(q) for q in g]);r=minimize_scalar(obj,bounds=(g[max(k-1,0)],g[min(k+1,len(g)-1)]),method='bounded');res.append((r.fun,name,en,r.x))
print('C1',sorted(res)[:28]);print('z range',z.min(0),z.max(0))
B=s['panel_b']; b=one(z)@ls(one(B['u']),B['v']); d=one(np.cos(2.4*z))@ls(one(np.cos(2.4*u)),p['v2']); Q=-np.expm1(-t[:,None]*[.41528,1.8129]);Y=np.c_[ls(Q,T.T).T,y];
def err(F):
 R=one(F); P=R@ls(R[:n],Y[:n]); return np.r_[np.mean(abs(P[:n]-Y[:n]),axis=0),np.mean(abs(P[n:72]-Y[n:]),axis=0),np.mean(abs(P[n:72,:2]@Q.T-T[n:]))]
choices=sorted(res)[:10]+[next(h for h in sorted(res) if h[2]=='lin')]; ans=[]
for _,fn,en,q in choices:
 f=Fams[fn];e=extras[en]; A=lambda w:one(np.c_[f(w,q),e(w)]);c=A(z)@ls(A(u),V)
 for l,H in {'C':np.c_[c,d],'arch':np.c_[z*np.roll(c,1,axis=1),c*d,c,d,c[:,0]*d[:,1]],'roll':np.c_[z*np.roll(c,1,axis=1),c*d,c,d],'cd':np.c_[c,d,allprod(c,d)],'cdd':np.c_[c,d,allprod(c,d),quad(d)],'c+z+d':np.c_[c,d,z],'ccdd':quad(np.c_[c,d]),'bd':np.c_[b,d,allprod(b,d)],'bcd':np.c_[b,c,d,allprod(b,c),c*d],'zcd':np.c_[z,c,d,allprod(z,c),c*d],'zquadcos':np.c_[z,quad(z),d],'zinv':np.c_[z,quad(z),d,cyc(z)/(1+z*z)]}.items():
  E=err(H);ans.append((np.mean(E[3:]),fn,en,l,E))
for row in sorted(ans,key=lambda a:a[0])[:24]:print(row)
