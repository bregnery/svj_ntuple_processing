import svj_ntuple_processing as svj
import numpy as np

variables = np.array(['girth', 'ptd', 'axismajor', 'axisminor', 'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi', 'weight', 'met', 'metphi', 'pt', 'eta', 'phi', 'e', 'mt', 'rt', 'leading_pt', 'leading_eta', 'leading_phi', 'leading_e', 'ak4_lead_eta', 'ak4_lead_phi', 'ak4_lead_pt', 'ak4_subl_eta', 'ak4_subl_phi', 'ak4_subl_pt'])

#mz = np.array(['150', '250', '350', '450', '550'])
mz = np.array(['350'])
#rinv = np.array(['0.1', '0.3', '0.7'])
rinv = np.array(['0.7'])

fermilab_files = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/orthogonalitystudy/HADD/madpt300_mz'

signal = {}
#sig_columns    = {}
for i in range(len(mz)):
   for j in range(len(rinv)):
      print(fermilab_files+str(mz[i])+'_mdark10_rinv'+str(rinv[j])+'.root')
      print('mz_'+mz[i]+'_rinv_'+rinv[j])
      arrays = svj.open_root(fermilab_files+str(mz[i])+'_mdark10_rinv'+str(rinv[j])+'.root')
      arrays = svj.filter_preselection(arrays)
      columns = svj.bdt_feature_columns(arrays)

      for k in range(len(variables)):
         signal[variables[k]] = columns.arrays[variables[k]]
      np.savez('mz_'+mz[i]+'_rinv_'+rinv[j]+'.npz', **signal)
