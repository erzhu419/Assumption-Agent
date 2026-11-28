import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; bank=s['sensor_calibration_sweeps']['single_bank_sweeps']
