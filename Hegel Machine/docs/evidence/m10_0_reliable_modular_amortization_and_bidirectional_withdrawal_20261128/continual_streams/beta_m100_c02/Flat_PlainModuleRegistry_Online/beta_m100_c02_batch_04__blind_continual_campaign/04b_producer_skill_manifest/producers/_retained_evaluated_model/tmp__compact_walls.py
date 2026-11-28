from fit_walls import *
k=s['kappa_x']; v=s['vel_x']; h=s['Lx']/(len(k)-1)
print('x h, robin, kappa, velocity',h,s['robin_x'],k,v)
print('y h, robin, kappa, velocity',s['Ly']/(ny-1),s['robin_y'],s['kappa_y'],s['vel_y'])
p=np.array([1.03554,1.03562,0,0,1.1371543,1.080679,1,1,0,0,0,0]); print('endpoint rates',p[:2]*k[[0,-1]]/h**2,p[4:6]*k[[0,-1]]/h**2,'increments', (p[:2]-1)*k[[0,-1]]/h**2,'wrap diffusivity',p[4:6]*k[[0,-1]])
base=np.r_[1.,1.,0,0,0,0,1.,1.,0,0,0,0]
forms={
 'x-only independent':(4,lambda t: [t[0],t[1],0,0,t[2],t[3]]),
 'x shared neighbor multiplier':(3,lambda t: [t[0],t[0],0,0,t[1],t[2]]),
 'x shared neighbor and wrap multipliers':(2,lambda t: [t[0],t[0],0,0,t[1],t[1]]),
 'x shared neighbor multiplier, equal absolute wraps':(2,lambda t: [t[0],t[0],0,0,*(t[1]*k[[0,-1]].mean()/k[[0,-1]])]),
 'x shared multiplier, wrap excess proportional to robin':(2,lambda t: [t[0],t[0],0,0,*(1+t[1]*s['robin_x']*h/k[[0,-1]])]),
 'x shared multiplier, wrap scaled face neighbors':(2,lambda t: [t[0],t[0],0,0,*(t[1]*(k[[0,-1]]+k[[1,-2]])/(2*k[[0,-1]]))]),
}
for name,(n,form) in forms.items():
 print(name)
 for hold in [-1,0,1,2]:
  mask=np.arange(3)!=hold
  def err(t): return predict(np.r_[form(t),base[6:]])-target
  f=least_squares(lambda t:err(t)[:,mask].ravel(),np.ones(n),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  e=err(f.x)
  print(hold,np.round(f.x,9),np.sqrt(np.mean(e**2,axis=0)).round(9))
