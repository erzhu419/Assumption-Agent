import numpy as np,pickle,itertools
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); lin=np.c_[np.ones(len(u)),u]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; err=np.linalg.norm(lin@fit(lin,v)-v,axis=1); good=err<4*np.median(err); u,v=u[good],v[good]; np.set_printoptions(precision=5,suppress=True)
funcs={'sin':lambda x,k:np.sin(k*x), 'tanh':lambda x,k:np.tanh(k*x), 'rat':lambda x,k:x/(1+(k*x)**2), 'cub':lambda x,k:x**3, 'none':lambda x,k:np.empty((len(x),0))}
xs=[(u[:,j],u[:,(j+1)%4],u[:,(j+2)%4],u[:,(j+3)%4]) for j in range(4)]
def design(j, typ, ks, ext=()):
 x,a,b,c=xs[j]; fs=[funcs[t] for t in typ];return np.c_[np.ones(len(u)),x,fs[0](x,ks[0]),fs[1](b,ks[1]),*[extf(x,a,b,c) for extf in ext]]
def result(typ,ks,ext=()):
 ds=[design(j,typ,ks,ext) for j in range(4)];cs=[fit(d,v[:,j]) for j,d in enumerate(ds)]; e=np.column_stack([d@c-v[:,j] for j,(d,c) in enumerate(zip(ds,cs))]);return np.mean(e**2)**.5,e,np.array(cs)
ans=[]
for typ in itertools.product(funcs,repeat=2):
 ks0=[1.0,2.15]; res=minimize(lambda k:result(typ,k)[0],ks0,bounds=[(.2,5),(.2,5)],method='Nelder-Mead'); f,e,c=result(typ,res.x);ans.append((f,typ,res.x,c))
for f,t,k,c in sorted(ans,key=lambda a:a[0])[:9]:print(t,k,f,c)
for t,k in [(('sin','sin'),[1,2.15]),(('cub','sin'),[1,2.15])]:
 f,e,c=result(t,k);print('2.15',t,f,c)
 for j,(x,a,b,c) in enumerate(xs):
  fs={'otherlin':a,'prevlin':c,'xy':x*a,'xp':x*c,'xb':x*b,'x2':x*x,'y2':a*a,'z2':b*b,'p2':c*c,'y*b2':a*b*b,'p*b2':c*b*b,'x2*b':x*x*b,'x*b2':x*b*b,'y*p':a*c,'sinx':np.sin(2.15*x),'cosb':np.cos(2.15*b),'sinmix':np.sin(2.15*(b+x)),'yp2':a*c*c,'y2p':a*a*c,'y2z':a*a*b}; print(j,sorted([(q,round(float(np.corrcoef(e[:,j],r)[0,1]),3)) for q,r in fs.items()],key=lambda q:-abs(q[1]))[:8])
