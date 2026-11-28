exec(open('/tmp/graph.py').read().split('F,Y,rates=amps(q)')[0]); np.set_printoptions(precision=4,suppress=True,linewidth=140)
a=panel_fit(u,p['v1'],1,'orth')[0](z); b=panel_fit(u,p['v2'],2,'orth')[0](z); dot=(a*b).sum(1); base=aff(np.c_[a,b]); ids=np.arange(len(y)); F,Y0,rates=amps(q); Y=np.c_[y,Y0];
models={'dot':np.c_[base,dot],'matched':np.c_[base,a*b],'matched+aa':np.c_[base,a*b,a*a,pair(a)],'matched+bb':np.c_[base,a*b,b*b,pair(b)],'dot+bb':np.c_[base,dot,b*b,pair(b)],'dot+z2':np.c_[base,dot,z*z,pair(z)],'dot+z':np.c_[base,dot,z,z*z,pair(z)],'dot+c2':np.c_[base,dot,z*z], 'dot+phi':np.c_[base,dot,(z+.45*z*z)**2], 'dot+formula':np.c_[base,dot, np.tanh(2.2*z)*(z+.45*z*z), z*z], 'dot+mix':np.c_[base,dot,*[a[:,i]*b[:,j] for i in range(4) for j in range(4)]], 'dot+gram':np.c_[base,dot,np.sum(a*a,axis=1),np.sum(b*b,axis=1)]}
for lab,m in models.items():
 co=ls(m,Y); ee=[]
 for va in [ids%6==k for k in range(6)]+[ids>=56,np.linalg.norm(z,axis=1)>=np.sort(np.linalg.norm(z,axis=1))[-14]]:
  tr=~va; T,Yt,_=amps(q[tr]); tar=np.c_[y[tr],Yt]; h=m[va]@ls(m[tr],tar);ee.append(np.mean(abs(np.c_[h[:,0]-y[va],h[:,1:]@T.T-q[va]]),0));
 e=np.array(ee); print(lab,'rms',np.std(m@co-Y,0),'CV',np.r_[e[:6,0].mean(),e[:6,1:].mean(),e[:6,0].max(),e[:6,1:].max()], 'stress',e[6:,0],e[6:,1:].max(1));
 if lab in ['dot','matched','dot+z','dot+formula','dot+gram']:print(co.T)
print('times',t,'fixed 1.8 .4 temporal noise',rms(q-(-np.expm1(-t[:,None]*[1.8,.4])@ls(-np.expm1(-t[:,None]*[1.8,.4]),q.T)).T)); print('amplitude range',Y.min(0),Y.max(0));
print('B coord relation'); pan=s['panel_b']; print(np.linalg.svd(ls(aff(pan['u']),pan['v'])[1:],compute_uv=False));
