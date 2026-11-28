import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
p=s['panel_a'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u];r=np.linalg.norm(v-h@ls(h,v),axis=1);k=r<3;print('A affine',ls(h[k],v[k]));
for j in range(4):
 x=u[k,j];p=u[k,(j+1)%4]; a=np.c_[np.ones(len(x)),x,p,x*p,x**3]; b=ls(a,v[k,j]);print('cal',j,b,'rms',np.std(a@b-v[k,j]))
p=s['panel_b'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u];print('B',ls(h,v))
p=s['panel_c'];u=p['u']; x=[u[:,j] for j in range(4)]; d={'one':np.ones(len(u))}; d.update({f'x{j}':w for j,w in enumerate(x)}); from itertools import combinations_with_replacement
for deg in (2,3):
 for ids in combinations_with_replacement(range(4),deg): d['p'+str(ids)]=np.prod(u[:,ids],axis=1)
for c in [0.22,.5,1,1.5,2.15,3]:
 for j,w in enumerate(x):
  for name,z in zip(['sin','cos','tanh','rat'],[np.sin(c*w),np.cos(c*w),np.tanh(c*w),w/(1+c*w*w)]): d[f'{name}{c}_{j}']=z
for key in ('v1','v2'):
 for j in range(4):
  y=p[key][:,j]; a=np.c_[np.ones(len(u)),u]; names=list(d)[5:]; F=np.column_stack([d[n] for n in names]); sel=[]
  for it in range(5):
   af=np.c_[a,F[:,sel]]; res=y-af@ls(af,y); ff=F-af@ls(af,F); sc=np.abs(ff.T@res)/np.maximum(np.linalg.norm(ff,axis=0),1e-9);sc[sel]=-1;sel.append(np.argmax(sc));
   ff=np.c_[a,F[:,sel]];print(key,j,it,names[sel[-1]],round(np.std(y-ff@ls(ff,y)),6),end='; ')
  print()
