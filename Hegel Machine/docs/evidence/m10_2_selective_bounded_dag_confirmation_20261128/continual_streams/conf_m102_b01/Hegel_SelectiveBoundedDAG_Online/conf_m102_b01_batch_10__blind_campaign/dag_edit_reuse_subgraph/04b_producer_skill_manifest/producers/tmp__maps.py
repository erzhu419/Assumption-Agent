exec(open('/tmp/probe.py').read().split('opts={')[0])
from scipy.optimize import minimize_scalar
pa=S['panel_a']; U,V=pa['u'],pa['v']; good=np.ones(len(U),dtype=bool); err=np.linalg.norm(aff(U)@ls(aff(U),V)-V,axis=1);good[np.argsort(err)[-3:]]=False
print('A ranges',U.min(0),U.max(0),'outliers',np.flatnonzero(~good));
def feats(x,kind,h):
 if kind=='sin': return np.sin(h[0]*x)
 if kind=='cos': return np.cos(h[0]*x)
 if kind=='sin2': return np.c_[np.sin(h[0]*x),np.cos(h[0]*x)]
 if kind=='tanh': return np.tanh(h[0]*x)
 if kind=='rat': return x/(1+h[0]**2*x*x)
 if kind=='atan': return np.arctan(h[0]*x)
 if kind=='sepsin': return x+h[1]*np.sin(h[0]*x)
 if kind=='sepodd': return x+h[1]*np.sin(h[0]*x)+h[2]*np.cbrt(x)
 if kind=='sepfull': return x+h[1]*np.sin(h[0]*x)+h[2]*np.cbrt(x)+h[3]*(1-np.cos(h[0]*x))
 if kind=='cube': return np.c_[x*x,x**3]
 if kind=='quint': return np.c_[x*x,x**3,x**4,x**5]
 if kind=='quadratic': return np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
 if kind=='pairs': return x*np.roll(x,-1,axis=1)
 if kind=='sinpair': return np.c_[np.sin(h[0]*x),x*np.roll(x,-1,axis=1)]
 if kind=='sinmix': return np.c_[np.sin(h[0]*x),np.sin(h[0]*(x+np.roll(x,-1,axis=1))),np.sin(h[0]*(x-np.roll(x,-1,axis=1)))]
def scan(a,b):
 out=[]
 for kind in ['sin','cos','sin2','tanh','rat','atan','sepsin','sepodd','sepfull','cube','quint','quadratic','pairs','sinpair','sinmix']:
  for base in ([False] if kind.startswith('sep') else [True,False] if kind in ['sin','tanh','rat','atan','sin2'] else [True]):
   def f(x,h): return aff(np.c_[x,feats(x,kind,h)] if base else feats(x,kind,h))
   def res(h): k=f(a,h);return (k@ls(k,b)-b).ravel()
   starts=[[s] for s in [1.25,2.15,2.9,4,5.5]] if kind not in ['cube','quint','quadratic','pairs'] else [[1]]
   if kind.startswith('sep'): starts=[[s,c]+[0]*(len({'sepsin':[0,0],'sepodd':[0,0,0],'sepfull':[0,0,0,0]}[kind])-2) for s in [1.25,2.15,2.9,4,5.5] for c in [-.4,.4]]
   rr=[least_squares(res,h,max_nfev=90) for h in starts];r=min(rr,key=lambda r:np.std(r.fun)); k=f(a,r.x); er=np.std(r.fun); press=np.std(r.fun.reshape(b.shape)/np.maximum(1-np.diag(k@np.linalg.pinv(k)),1e-9)[:,None]); print(kind,base,np.round(r.x,5),er,press);out.append((press,kind,base,r.x,ls(k,b)))
 return sorted(out,key=lambda m:m[0])
if __name__=='__main__':
 print('A'); aa=scan(U[good],V[good]);print('B'); bb=scan(u,v);import pickle;pickle.dump((aa,bb),open('/tmp/mapfits','wb'))
 for j in range(4):
  a=U[good];y=V[good,j];x=a[:,j];k=a[:,(j+1)%4];
  for variant in ['periodic','rational','tanh']:
   def design(h):
    if variant=='periodic': t=np.sin(h[0]*x);r=np.cos(h[0]*x);l=np.sin(h[0]*k)
    elif variant=='rational': t=x/(1+h[0]**2*x*x);r=x*t;l=k/(1+h[0]**2*k*k)
    else:t=np.tanh(h[0]*x);r=t*t;l=np.tanh(h[0]*k)
    return np.c_[aff(a),x*k,x**3,t,r,l]
   def res(h): f=design(h);return f@ls(f,y)-y
   fit=min((least_squares(res,[i],max_nfev=120) for i in [1,2,3,4,5,6,7,9]),key=lambda r:np.std(r.fun)); print('A channel',j,variant,fit.x,np.std(fit.fun),np.round(ls(design(fit.x),y),4))
