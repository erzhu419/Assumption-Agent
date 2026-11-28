exec(open('/tmp/cv.py').read().split("for name,u in")[0])
np.set_printoptions(precision=4,suppress=True,linewidth=140)
c,d=branches(z); cross=c[:,0]*d[:,1]; dot=(c*d).sum(1)
zc=outer(z,c); bc=outer(b,c); cubic=np.column_stack([np.prod(z[:,p],axis=1) for p in combinations(range(4),3)]+[z[:,i]**2*z[:,(i+1)%4] for i in range(4)]+[z[:,i]*z[:,(i+1)%4]**2 for i in range(4)])
base=np.c_[c,d,dot,cross]; Y=np.c_[y,T];
for label,h in [('base',base),('z+zc',np.c_[base,z,zc]),('z+bc',np.c_[base,z,bc]),('cubic',np.c_[base,z,cubic]),('minimal',np.c_[base,z,zc[:,[3,4,8,9,14]]]),('all cd',np.c_[c,d,z,cubic,outer(c,d)]),('z*c cyclic',np.c_[base,z,zc[:,[0,5,10,15]]]),('b*c cyclic',np.c_[base,z,bc[:,[0,5,10,15]]])]:
 score(label,h); H=_one(h); print('rank/rmse',np.linalg.matrix_rank(H),np.sqrt(np.mean((H[:n]@_fit(H[:n],Y[:n])-Y[:n])**2,0))); print('weights [y, slow, medium, fast]\n',_fit(H[:n],np.c_[y,A][:n]));
 print('time-stage ranks',np.linalg.svd(H[:n]@_fit(H[:n],T[:n]),compute_uv=False));
