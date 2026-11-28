import pickle, numpy as np
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
def calibrate(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; g=np.max(abs(v-u),axis=1)<3; u,v=u[g],v[g]; I=np.eye(4); N=np.roll(I,1,axis=1)
 b=ls(np.c_[np.tile(I,(len(u),1)),(u[:,:,None]*I).reshape(-1,4),u.ravel()**3,((u-1)*np.roll(u,-1,axis=1)).ravel()],v.ravel()); o,a,c,h=np.split(b,[4,8,9]); w=(x-o)/a
 for _ in range(20):
  r=np.roll(w,-1,axis=1); e=o+a*w+c*w**3+h*(w-1)*r-x; J=(a+3*c*w*w+h*r)[:,:,None]*I+(h*(w-1))[:,:,None]*N
  w-=np.clip(np.linalg.solve(J,e[...,None])[...,0],-.5,.5)
 return w
F={
 'sin':lambda u,p:aff(np.sin(p[0]*u)), 'tanh':lambda u,p:aff(np.tanh(p[0]*u)), 'atan':lambda u,p:aff(np.arctan(p[0]*u)),
 'sin+cos':lambda u,p:np.c_[aff(np.sin(p[0]*u)),np.cos(p[0]*u)], 'sin+sq':lambda u,p:np.c_[aff(np.sin(p[0]*u)),u*u],
 'tanh+sq':lambda u,p:np.c_[aff(np.tanh(p[0]*u)),u*u], 'x+cos':lambda u,p:np.c_[aff(u),np.cos(p[0]*u)], 'x+sin+cos':lambda u,p:np.c_[aff(u),np.sin(p[0]*u),np.cos(p[0]*u)],
 'sin+cos2':lambda u,p:np.c_[aff(np.sin(p[0]*u)),np.cos(p[1]*u)], 'tanh+cos':lambda u,p:np.c_[aff(np.tanh(p[0]*u)),np.cos(p[1]*u)],
 'x+sin':lambda u,p:np.c_[aff(u),np.sin(p[0]*u)],'sin+x+sq':lambda u,p:np.c_[aff(u),np.sin(p[0]*u),u*u],
 'rat':lambda u,p:aff(u**3/(1+p[0]*abs(u)**p[1])+p[2]*u),
 'tanh+x':lambda u,p:np.c_[aff(u),np.tanh(p[0]*u)],'x+exp':lambda u,p:np.c_[aff(u),np.exp(p[0]*u)],
 'x+cos+xsin':lambda u,p:np.c_[aff(u),np.cos(p[0]*u),u*np.sin(p[0]*u)]}
def models(u,v):
 for n,f in F.items():
  def r(p):
   a=f(u,p); return (a@ls(a,v)-v).ravel()
  starts=[[a] for a in (.5,1.,1.8,2.8,4.)]; lo=[.03]; hi=[6.]
  if n in ('sin+cos2','tanh+cos'): starts=[[a,b] for a in (.7,1.5,2.8) for b in (.7,1.5,2.8)];lo*=2;hi*=2
  if n=='rat': starts=[[.5,2.5,-.1],[1.3,2,-.1]];lo=[.015,.6,-4];hi=[8,6,4]
  m=min((least_squares(r,x,bounds=(lo,hi),max_nfev=110) for x in starts),key=lambda o:o.cost); a=f(u,m.x); h=np.diag(a@np.linalg.pinv(a)); er=np.sqrt(np.mean((r(m.x).reshape(v.shape)/(1-h[:,None]))**2)); yield er,n,m.x,ls(a,v)
if __name__=='__main__':
 x=calibrate(S,S['train']['x']); y=S['train']['y']; t=S['train']['traj']; print('coordinate spans',x.min(0),x.max(0),'archive-scale PCA',np.linalg.svd(t,compute_uv=False))
 saved={}
 for name,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=S[name]; mods=sorted(models(p['u'],p[key])); saved[key]=mods
  for er,n,w,b in mods[:10]: print(key,n, np.round(w,5),round(er,7))
 pickle.dump(saved,open('/tmp/op04.pkl','wb'))
 def cv(z):
  a=aff(z); inter=z[:,0]*z[:,2]+z[:,1]*z[:,3]; ds=[a,np.c_[a,inter],np.c_[a,z[:,0]*z[:,2],z[:,1]*z[:,3]]]; print(' / '.join(str(np.round([np.mean(abs((d@ls(d[:56],y[:56])-y)[56:])),np.mean(abs((d@ls(d[:56],t[:56])-t)[56:]))],5)) for d in ds),end='')
 print('raw',end=' ');cv(x);print()
 for k,mods in saved.items():
  for er,n,p,b in mods[:5]: print(k,n,end=' ');cv(F[n](x,p)@b);print()
