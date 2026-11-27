exec(open('/tmp/wall_diag.py').read().split("F={'base'")[0])
np.set_printoptions(precision=5,suppress=True)
ridge=s['source_ridge']; P=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(nz),z.T); yy=y-z@(P@y)
def certificate(wall):
 S,U=m._response(s,wall,src); A=S-z@(P@S); w=m._weights(A,yy,ridge,cap,total); r=A@w-yy; G=A.T@r+ridge*w; J=(r@r+ridge*(w@w))/2; return w,w@U.reshape(n,-1),G,J,A
w0,f0,g0,j0,A0=certificate(np.ones(8)); print('nominal solution',w0,'objective',j0,'gradient',g0)
_,e0,q0=fit(()); print('nominal empirical residual by shift (RMS then mean):',np.sqrt(np.mean(e0.reshape(3,-1)**2,axis=1)),e0.reshape(3,-1).mean(1)); print('nominal empirical residual by routing bank:',np.sqrt(np.mean(e0.reshape(-1,4)**2,axis=0))); print('cross-shift empirical residual corr:',np.corrcoef(e0.reshape(3,-1)))
for name,groups in [('shared robin',((0,1,4,5),)),('axis robin',((0,1),(4,5))),('separate robin',((0,),(1,),(4,),(5,))),('all',tuple((j,) for j in range(8)))]:
 wall,e,q=fit(groups); w,f,g,j,A=certificate(wall); d=g0-m._dual(g0,w0,cap); gap_ref=np.maximum(d,0)@w+np.maximum(-d,0)@(cap-w)+.5*np.linalg.norm(A0@(w-w0))**2+.5*ridge*np.linalg.norm(w-w0)**2
 print(name,'relative S change',np.linalg.norm(A-A0)/np.linalg.norm(A0),'weight max shift',np.max(abs(w-w0)),'field rel shift',np.linalg.norm(f-f0)/np.linalg.norm(f0),'objective',j,'nominal objective excess',gap_ref)
 print('  held-shift RMS',np.array([np.sqrt(np.mean(fit(groups,np.arange(len(y))//32!=k)[1][k*32:(k+1)*32]**2)) for k in range(3)]))
print('nominal held-shift RMS',np.array([np.sqrt(np.mean(fit((),np.arange(len(y))//32!=k)[1][k*32:(k+1)*32]**2)) for k in range(3)]))
print('nominal baseline wall-tangent distinguishability')
S0=m._response(s,np.ones(8),src)[0]; T=np.c_[S0[:,1:]-S0[:,:1],z]; D=[]
for j in range(8):
 wall=np.ones(8); wall[j]+=1e-3; D.append((m._response(s,wall,src)[0]-S0)@q0[:n]/1e-3)
D=np.array(D).T; D-=T@np.linalg.lstsq(T,D,rcond=None)[0]; print(np.linalg.svd(D,compute_uv=False)); print(np.corrcoef(D.T))
