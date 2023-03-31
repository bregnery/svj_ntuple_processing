import svj_ntuple_processing as svj
import numpy as np

variables = np.array(['leading_eta', 'leading_phi', 'leading_pt', 'ak15_lead_e', # leading ak15jets
		      'eta', 'phi', 'rt', 'pt', 'mt', 'ak15_subl_e', 'metdphi',  # sub-leading ak15jets
                      'ecfc2b1', 'girth', 'ptd', 'axismajor', 'axisminor', #sub-leading ak15jets substructure variables
		      'met', 'metphi', # met related variables
                      'ak4_subl_pt', 'ak4_subl_eta', 'ak4_subl_phi', 'ak4_lead_pt', 'ak4_lead_eta', 'ak4_lead_phi', 'ak8_lead_pt' #other general variables
		     ]) 

mz = np.array(['200', '250', '300', '350', '400', '450', '500', '550'])
mdark = np.array(['1', '5', '10'])
rinv= np.array(['0.3'])

files = ['/mnt/hadoop/cms/store/user/snabili/SIG/mDark'+str(i)+'/TREEMAKER/HADD/madpt300_mz' for i in mdark]

signal = {}
for i in range(len(mz)):
   for l in range(len(rinv)):
      for j in range(len(mdark)):
         print(mz[i], rinv[l], mdark[j])
         arrays = svj.open_root(files[j]+str(mz[i])+'_mdark'+str(mdark[j])+'_rinv'+str(rinv[l])+'.root')
         arrays = svj.filter_preselection(arrays)
         columns = svj.bdt_feature_columns(arrays)
         
         for k in range(len(variables)):
            signal[variables[k]] = columns.arrays[variables[k]]
         signal['total'] = arrays.cutflow['raw']
         signal['preselection'] = arrays.cutflow['preselection']
         signal['cutflow'] = arrays.cutflow
         np.savez('/home/snabili/data/svj_ntuple_processing/sig_files/hadd/madpt300_mz'+mz[i]+'_mdark'+str(mdark[j])+'_rinv'+str(rinv[l])+'.npz', **signal)
