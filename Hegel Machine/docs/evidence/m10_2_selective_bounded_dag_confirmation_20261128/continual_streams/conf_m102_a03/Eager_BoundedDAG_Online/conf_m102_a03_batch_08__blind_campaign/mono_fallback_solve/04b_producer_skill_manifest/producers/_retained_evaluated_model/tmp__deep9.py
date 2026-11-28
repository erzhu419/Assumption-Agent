exec(open('/tmp/graph7.py').read().split('for dg in (1,2,3):')[0]); from scipy.optimize import least_squares
from itertools import combinations
p=s['panel_b']; U,V=p['u'],p['v'];
# Undo a possible output mixing matrix and inspect each independent source.
for i in range(4):
 a=U[:,i]; F=np.c_[np.ones(len(U)),V]; H=np.c_[a,a**2,a**3,a**4,a**5,a**7]; R=H-F@ls(F,H); _,sv,vh=np.linalg.svd(R,full_matrices=False); co=vh[-1];print('source',i,sv,co/np.max(abs(co)))
forms={
'cube': (lambda x,w:x+w[0]*x**3,[.4]),
'poly': (lambda x,w:x+w[0]*x**3+w[1]*x**5+w[2]*x**7,[.4,-.1,.02]),
'ratcube': (lambda x,w:(x+w[0]*x**3)/(1+np.exp(w[1])*x*x),[1,-1]),
'ratpure': (lambda x,w:x**3/(1+np.exp(w[0])*x*x),[-1]),
'ratsq': (lambda x,w:(x+w[0]*x**3)/(1+np.exp(w[1])*x*x)**2,[1,-1]),
'shift': (lambda x,w:((x+w[2])+w[0]*(x+w[2])**3)/(1+np.exp(w[1])*(x+w[2])**2),[1,-1,0]),
'rat4': (lambda x,w:(x+w[0]*x**3)/(1+np.exp(w[1])*x*x+np.exp(w[2])*x**4),[1,-1,-2]),
'arctan': (lambda x,w:x+w[0]*np.arctan(np.exp(w[1])*x),[-.6,.3]),
'sin': (lambda x,w:x+w[0]*np.sin(np.exp(w[1])*x),[-.4,1]),
'rattele': (lambda x,w:x/(1+.22*x*x),[]),
'rcut': (lambda x,w:x**3/(1+.22*x*x),[]),
}
best=[]
for tag,(fun,ini) in forms.items():
 for extra in ('','lin','cyc','sq','other'):
  def feat(x,w):
   h=fun(x,w); return aff(np.c_[h,*{'':[], 'lin':[x], 'cyc':[x*roll(x)],'sq':[x*x],'other':[x, x*roll(x),x*x]}[extra]])
  def err(w):
   H=feat(U,w);return (H@ls(H,V)-V).ravel()
  w=least_squares(err,ini,bounds=(-10,10),max_nfev=100).x if ini else []
  H=feat(U,w); er=err(w).reshape(V.shape); lev=np.sum(H*np.linalg.pinv(H).T,axis=1); cv=np.mean(abs(er/(1-np.minimum(lev,.999))[:,None]));
  h=feat(z,w)@ls(H,V); F=PolynomialFeatures(2).fit_transform(h); e=score(F);best.append((cv,np.std(er),tag,extra,np.round(w,6),e[0],e[1:].mean()));D[tag+extra]=h
print('panel best',sorted(best,key=lambda a:a[0])[:16]);print('readout best',sorted(best,key=lambda a:a[-2])[:12]);
for tag in ['B(z)','ratcube','ratcube lin'.replace(' ',''),'poly','rcutlin']:
 if tag not in D:continue
 h=D[tag]; F=PolynomialFeatures(2).fit_transform(h); c=ls(F[:52],Y[:52]); r=Y-F@c; print('readout',tag,'in/out',np.mean(abs(r[:52]),0)[[0,1,8]],np.mean(abs(r[52:]),0)[[0,1,8]]); print('coef',np.round(c[:,[0,1,8]],3).T)
# Is temporal residual governed by two separable bases?
t=s['eval_times']; T=Y[:,1:]; mu=T[:52].mean(0); _,sing,Vh=np.linalg.svd(T[:52]-mu,full_matrices=False);print('traj',sing,mu,Vh[:3]);
