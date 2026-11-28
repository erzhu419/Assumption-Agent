exec(open('/tmp/refine11.py').read().split('for base,gate,reg')[0])
from scipy.optimize import least_squares
u=s['panel_b']['u']; v=s['panel_b']['v']; h0=np.array([-.52,-.15,1,-.12]);
for name,U in [('b',u),('c',s['panel_c']['u'])]:
 for k,V in s['panel_'+name].items():
  if k=='u': continue
  W=m._ls(m._aff(U),V); print(name,k,'affine weights',W.round(4),'rms',np.std(m._aff(U)@W-V),flush=True)
for model in ['free','tanh','cos','tanh_cos','tanh_aff_cos','sin_aff','tanh2','tanh2aff','gatequad','satraw']:
 def fun(p):
  h=x[:,2]+L@p[:3]+p[3]; k=p[4]; t=np.tanh(k*h); c=np.cos(1.25*h); h2=x@p[5:9]+p[9]; sec=np.tanh(h2); G={ 'free': A*t[:,None], 'tanh': t[:,None], 'cos':np.c_[t,c], 'tanh_cos':np.c_[t,c,np.sin(1.25*h)],'tanh_aff_cos':np.c_[A*t[:,None],c], 'sin_aff':np.c_[A*np.sin(k*h)[:,None]],'tanh2':np.c_[t,sec], 'tanh2aff':np.c_[A*t[:,None],sec], 'gatequad':np.c_[t,h*t,t*t], 'satraw':np.c_[A*t[:,None],np.tanh(2.15*x)]}[model]; return np.c_[Q,G]
 E=[]; pars=[]; d=fun(np.r_[h0[[0,1,3]],0,1.8,np.ones(5)]).shape[1]; R=np.diag(np.r_[np.zeros(5),np.full(10,.003),np.full(d-15,.008)])
 for j,idx in enumerate([I]+splits):
  a,b=(idx,idx) if j==0 else (idx[:52],idx[52:]); Y=np.r_[y[a],np.zeros(d)];
  def calc(p):
   p=np.r_[p,np.zeros(10-len(p))]; D=fun(p); D0=np.r_[D[a],R]; w=m._ls(D0,Y); return D0@w-Y,D[b]@w,w
  lo=[-3]*4+[.2]; hi=[3]*4+[6]; starts=[[-.52,-.15,-.12,0,1.8],[-.5,-.15,-.12,-.5,2.15],[0,0,0,0,1.25]]
  if 'tanh2' in model: lo += [-5]*5; hi += [5]*5; starts=[p+h for p in starts[:2] for h in [[1,0,-1,0,0],[0,1,0,-1,0],[-.5,-.15,1,-.12,0]]]
  r=min([least_squares(lambda p:calc(p)[0],p,bounds=(lo,hi),max_nfev=100) for p in starts],key=lambda t:t.cost); _,pred,w=calc(r.x); E.append(np.mean(abs(pred-y[b]))); pars.append(r.x.round(3));
  if j==0: print(model,'full',pars[-1],w.round(3),flush=True)
 E=np.array(E); print(model,'MAE',E[[0,1,-4,-3,-2,-1]].round(5),'rand',E[2:14].max().round(5),'tail pars',pars[-2:],flush=True)
