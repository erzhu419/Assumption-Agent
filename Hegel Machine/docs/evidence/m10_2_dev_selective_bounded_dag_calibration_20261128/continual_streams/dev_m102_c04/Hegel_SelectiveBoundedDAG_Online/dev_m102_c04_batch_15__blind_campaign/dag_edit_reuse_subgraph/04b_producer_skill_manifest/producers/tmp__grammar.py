exec(open('/tmp/spatial_probe.py').read().split('pa=s[')[0])
from itertools import permutations

def groups(x):
 a,b=x[:,[i for i,j in tri]],x[:,[j for i,j in tri]]
 return {'u':x,'sum':a+b,'dif':a-b,'pair':a*b,'trip':trip(x),'rad':a*a+b*b,'sqdif':a*a-b*b}

def scan1(u,v,fn,lo,hi):
 def obj(k):
  a=np.c_[affine(u),fn(u,k)];return np.mean((a@fit(a,v)-v)**2)
 ks=np.arange(lo,hi+.0001,.1); j=np.argmin([obj(k) for k in ks]); m=minimize_scalar(obj,bounds=(ks[max(j-1,0)],ks[min(j+1,len(ks)-1)]),method='bounded'); return np.sqrt(m.fun),m.x

def sweep(u,v):
 out=[]
 funs={'sin':lambda z,k:np.sin(k*z),'cos':lambda z,k:np.cos(k*z),'sqrat':lambda z,k:z*z/(1+k*z*z),'rat':lambda z,k:z/(1+k*z*z),'absrat':lambda z,k:z/(1+k*abs(z)),'tanh':lambda z,k:np.tanh(k*z)}
 for g in groups(u):
  for f,op in funs.items():out.append((*scan1(u,v,lambda x,k:op(groups(x)[g],k),.05,5.05),g+' '+f))
 a,b=u[:,[i for i,j in tri]],u[:,[j for i,j in tri]]
 for name,den in {'rad':a*a+b*b,'sum2':(a+b)**2,'dif2':(a-b)**2,'prod2':(a*b)**2,'absprod':abs(a*b),'absdiff':abs(a-b),'abssum':abs(a+b),'global':(u*u).sum(1)[:,None],'a2':a*a,'b2':b*b,'abs_a':abs(a),'abs_b':abs(b)}.items():
  for tag,num in [('prod',a*b),('diffsq',a*a-b*b),('sum',a+b)]:out.append((*scan1(u,v,lambda x,k:num/(1+k*den),.05,4.05),tag+'/'+name))
  if name in ('a2','abs_a'):out.append((*scan1(u,v,lambda x,k:np.c_[a*b/(1+k*den),a*b/(1+k*(b*b if name=='a2' else abs(b)))],.05,4.05),'directed prod/'+name))
 for g in ['sum','dif','pair']:
  out.append((*scan1(u,v,lambda x,k:np.c_[np.sin(k*x),np.sin(k*groups(x)[g])],.05,5.05),'u+'+g+' sine'))
 print(*sorted(out)[:17],sep='\n')
pa=s['panel_a'];m=np.linalg.norm(pa['v']-pa['u'],axis=1)<4
print('A');sweep(pa['u'][m],pa['v'][m]);print('C2');sweep(s['panel_c']['u'],s['panel_c']['v2'])
