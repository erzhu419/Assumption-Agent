exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
C,E=profile(tau); print('fixed amplitudes:')
a,rr=m._design(tau,xd,pd,hd,dt,t); gap=rr[:,:2]*tau; r=rr[:,2:]@beta[2:]; z=hd[:,None]-gap
np.set_printoptions(precision=4,suppress=True)
print('rows: h z0 z1 drift A0 A1 D')
print(np.c_[hd,z,r,C]); print('rate vs curve mode error',np.std(E,axis=0))
# Compare amplitudes using algebraic features, then inspect signs/ratios to identify the chamber law.
sets=[('hold',np.c_[hd]),('hold,state',np.c_[hd,z]),('hold,history',np.c_[hd,pd]),('hold,drift',np.c_[hd,r]),('hold,state,drift',np.c_[hd,z,r]),('hold,state,features',np.c_[hd,z,xd]),('state,drift cubic',np.c_[hd,z,r,r*r,r**3]),('hold,drift cubic',np.c_[hd,r,r*r,r**3])]
for name,F in sets:
 F=np.c_[np.ones(len(F)),F]; b=np.linalg.lstsq(F,C,rcond=None)[0]; H=np.sum(np.linalg.qr(F)[0]**2,axis=1); er=(C-F@b)/(1-H[:,None]); Q=er@basis(tau).T
 print(name,'coef',b.T,'amp LOO RMS',np.sqrt(np.mean(er**2,axis=0)),'LOO curve/rate max',np.max(abs(Q[:,:-1])),np.max(abs(Q[:,-1])))
# Separate fast varying response and slow effective drift: report the nonlinear transform in residual coordinates.
print('A0 - theta0 h, A1 - theta1 gap1, D - drift',np.c_[C[:,0]-beta[0]*hd,C[:,1]-beta[1]*gap[:,1],C[:,2]-r]); print('labs',di['lab_id'])
