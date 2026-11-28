exec(open('/tmp/explore15.py').read().split('for j in (1,2,3):')[0])
print('affine',ls(aff(u),v),rm(aff(u)@ls(aff(u),v)-v)); print(np.round(np.c_[u,v][:12],3))
fn={'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'root':lambda x,k:x/np.sqrt(1+k*x*x),'sat':lambda x,k:x/(1+k*abs(x)), 'rat':lambda x,k:x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'pow':lambda x,k:x*abs(x)**k,'sinh':lambda x,k:np.sinh(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'exp':lambda x,k:np.exp(-k*x*x)}
R=np.roll(u,-1,axis=1); B=aff(u); pair=np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]); dr=[('u',u),('u+r',u+R),('u-r',u-R),('u*r',u*R)]; out=[]
for nm,g in dr:
 for name,f in fn.items():
  def fit(k):
   d=aff(u,f(g,k[0]));w=ls(d,v);return (d@w-v).ravel(),w
  z=min((least_squares(lambda k:fit(k)[0],[k],bounds=(.06,8)) for k in (1.25,2.15,4.0)),key=lambda p:p.cost);out.append((rm(z.fun),nm,name,z.x[0],fit(z.x)[1]))
print('a best');
for er,nm,name,k,w in sorted(out,key=lambda x:x[0])[:12]: print(er,nm,name,k, np.round(w,3) if er<.03 else '')
print('A poly');
for j in (1,2,3):
 r=np.roll(u,j,axis=1)
 for abc in [(u*r,u**3),(u*r,u**3,r**3),(u**2,r**2,u*r),(u**2,r**2,u*r,u**3,r**3,u*u*r,u*r*r)]:
  d=aff(u,*abc); print(j,d.shape,rm(d@ls(d,v)-v))
