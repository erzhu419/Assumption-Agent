import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c03_batch_03_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=m._calibrate(s['panel_a'],s['train']['x']); u=s['panel_c']['u']; v=s['panel_c']['v2']; n=52; t=s['eval_times']; Y=s['train']['traj']; T=np.c_[s['train']['y'],Y]; ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]; c=m._branch(s['panel_c'],z,'v1','soft')
fun={'gauss':lambda x,r:np.exp(-(x*r)**2),'cos':lambda x,r:np.cos(r*x),'rat':lambda x,r:1/(1+(r*x)**2),'soft':lambda x,r:x/(1+r*abs(x)),'tanh':lambda x,r:np.tanh(r*x),'sin':lambda x,r:np.sin(r*x),'sq':lambda x,r:x*x,'cube':lambda x,r:x**3,'abs':lambda x,r:abs(x),'none':lambda x,r:x}
cross=lambda x,y:(x[:,:,None]*y[:,None,:]).reshape(len(x),-1)
cyc=lambda x,j:x*np.roll(x,-j,axis=1)
base=[('linear',lambda x:m._aff(x)),('adj',lambda x:np.c_[m._aff(x),cyc(x,1)]),('opp',lambda x:np.c_[m._aff(x),cyc(x,2)[:,:2]]),('pairs',lambda x:np.c_[m._aff(x),cyc(x,1),cyc(x,2)[:,:2]])]
res=[]
for bn,bf in base:
 for fn,ff in fun.items():
  def des(x,r):return np.c_[bf(x),ff(x,r)]
  def prof(r): d=des(u,r);return (d@ls(d,v)-v).ravel()
  for dim in (1,4):
   opts=[least_squares(prof,np.full(dim,r),bounds=(.075,5),gtol=1e-12,max_nfev=80) for r in (.4,1.2,2.8)] if fn not in ('sq','cube','abs','none') else [least_squares(prof,[1.])]
   opt=min(opts,key=lambda o:o.cost); r=opt.x; d=des(z,r)@ls(des(u,r),v); errors=[]
   for f in [np.c_[m._aff(z),c,d,cross(c,d)],np.c_[m._aff(z),c,d,c*d],np.c_[m._aff(z),c,d,(c*d).sum(1)]]:
    p=f@ls(f[:n],T[:n]); er=abs(p-T); errors.append(tuple(np.round([er[:n,0].mean(),er[n:,0].mean(),er[n:,1:].mean()],5)))
   res.append((np.mean(opt.fun**2)**.5,bn,fn,dim,np.round(r,5),errors))
for row in sorted(res,key=lambda a:a[0])[:24]:print(row)
print('best by response');
for row in sorted(res,key=lambda a:max(a[-1][0][1:]))[:12]: print(row)
print('times',t)
