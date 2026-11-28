import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
D=lambda z:np.c_[np.ones(len(z)),z]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def AF(u):return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
def cal(x):
 u,v=s['panel_a'].values();k=np.ones(len(u),bool)
 for _ in range(6):
  c=fit(AF(u)[k],v[k]); r=np.linalg.norm(AF(u)@c-v,axis=1);k=r<max(.002,3*np.median(r));
 c=fit(AF(u)[k],v[k]); z=np.linalg.solve(c[1:5].T,(x-c[0]).T).T
 for _ in range(10):
  j=c[1:5]+np.roll(z,-1,axis=1)[:,:,None]*c[5:9]+np.roll(z[:,:,None]*c[5:9],1,axis=1)+3*z[:,:,None]**2*c[9:]
  z-=np.linalg.solve(j.transpose(0,2,1),(AF(z)@c-x)[...,None])[...,0]
 return z,c,r
forms={
 'power':(lambda x,a,b:np.sign(x)*abs(x)**a+b*x,[2.,0],([.1,-8],[6,8])),
 'dead':(lambda x,a,b:np.sign(x)*np.maximum(abs(x)-a,0)**b,[.3,1.4],([0,.1],[1,5])),
 'erf':(lambda x,a,b:erf(a*x)+b*x,[1.,0],([.05,-8],[8,8])),
 'tanh':(lambda x,a,b:np.tanh(a*x)+b*x,[1.,0],([.05,-8],[8,8])),
 'sin':(lambda x,a,b:np.sin(a*x)+b*x,[1.,0],([.05,-8],[8,8])),
 'rat':(lambda x,a,b:x/(1+a*x*x)+b*x,[1.,0],([.001,-8],[12,8])),
 'soft':(lambda x,a,b:x/(1+a*abs(x))+b*x,[1.,0],([.001,-8],[12,8])),
 'clip':(lambda x,a,b:np.clip(x,-a,a)+b*x,[.5,0],([.05,-8],[1.8,8])),
 'asinh':(lambda x,a,b:np.arcsinh(a*x)+b*x,[1.,0],([.05,-8],[12,8])),
 'expquad':(lambda x,a,b:np.expm1(a*x)+b*x,[.45,0],([.01,-8],[3,8])),
 'sqrtquad':(lambda x,a,b:np.sqrt(1+a*x*x)+b*x,[.45,2],([.001,-8],[12,8])),
 'quadlog':(lambda x,a,b:x+a*x*x+b*x**3,[.45,0],([-3,-3],[3,3])),
 'p-rat':(lambda x,a,b:x/(1+a*x)+b*x,[.25,0],([-.45,-8],[.45,8])),
 'cos':(lambda x,a,b:np.cos(a*x)+b*x,[1.,1],([.05,-8],[8,8])),
 'relusq':(lambda x,a,b:x+b*np.maximum(x-a,0)**2,[0,.5],([-1,-5],[1,5])),
}
def panel(u,v):
 res=[]
 for key,(f,q,bounds) in forms.items():
  def err(q):
   h=D(f(u,*q));return (h@fit(h,v)-v).ravel()
  qs=[least_squares(err,[a,q[1]],bounds=bounds,max_nfev=160) for a in np.clip([q[0],.2,.45,.85,1.8,3.5],bounds[0][0]+1e-5,bounds[1][0]-1e-5)]
  p=min(qs,key=lambda p:p.cost); res.append((np.std(p.fun),key,p.x,fit(D(f(u,*p.x)),v)))
 return sorted(res,key=lambda r:r[0])
if __name__=='__main__':
 z,c,r=cal(s['train']['x']); print('A',c.round(5),'residuals',np.sort(r).round(5)); print('z range',z.min(0),z.max(0));
 for u,v in [tuple(s['panel_b'].values()),(s['panel_c']['u'],s['panel_c']['v2'])]:
  print('u range',u.min(0),u.max(0)); print('best models',[(round(a,7),b,c.round(6)) for a,b,c,d in panel(u,v)]); print('vsvd',np.linalg.svd(v-v.mean(0),compute_uv=False))
  m=fit(D(u),v); w=(v-m[0])@np.linalg.inv(m[1:]); print('u, inverse-affine output',np.c_[u,w].round(3))
