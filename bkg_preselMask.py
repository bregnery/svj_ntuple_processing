#========================================================================================
# submit_bkg_preselMask.py --------------------------------------------------------------
#----------------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Thomas Klijnsma -------------------------------------------
#----------------------------------------------------------------------------------------
# Template for submitting HT condor jobs with submit_bkg_preselMask.py ------------------
#----------------------------------------------------------------------------------------

# Imports
import os, os.path as osp, re, traceback
from jdlfactory_server import data, group_data # type: ignore
import seutils # type: ignore
import svj_ntuple_processing as svj # type: ignore

seutils.set_preferred_implementation(group_data.storage_implementation.encode())

#----------------------------------------------------------------------------------------
# Program specific functions
#----------------------------------------------------------------------------------------

def dst(path):
    """
    Generates a destination .npz for a rootfile
    """
    # Get the stump starting from the dir with year in it
    path = '/'.join(path.split('/')[-3:])
    path = path.replace('.root', '_presel_mask.npz')
    return group_data.stageout + ('' if group_data.stageout.endswith('/') else '/') + path

def save_mask(presel_mask, train_val_test_mask, outfile):
    do_stageout = False
    if seutils.path.has_protocol(outfile):
        remote_outfile = outfile
        outfile = svj.uid() + '.npz'
        do_stageout = True

    logger.info('Dumping to %s', outfile)

    # Automatically create parent directory if not existent
    outdir = osp.dirname(osp.abspath(outfile))
    if not osp.isdir(outdir):
        os.makedirs(outdir)

    np.savez(
        outfile,
        presel_mask = presel_mask,
        train_val_test_mask = train_val_test_mask
        )

    if do_stageout:
        logger.info('Staging out %s -> %s', outfile, remote_outfile)
        seutils.cp(outfile, remote_outfile)
        os.remove(outfile)

def preselection_mask(array, single_muon_cr=False):
    """Apply the preselection on the array.

    Args:
        single_muon_cr (bool): If true, *selects* a muon instead of applying the lepton
            veto. Disables the AK8Jet.pT cut and the triggers, since this mode is used
            to study the trigger efficiencies.
    """
    copy = array.copy()
    a = copy.array
    cutflow = copy.cutflow
    
    if not single_muon_cr:
        # AK8Jet.pT>500
        #mask = (ak.count(a['JetsAK8.fCoordinates.fPt'], axis=-1) >= 1)  & (a['JetsAK8.fCoordinates.fPt'][:,0] > 500.)  # at least one jet and leading>500
        mask = (ak.count(a['JetsAK8.fCoordinates.fPt'], axis=-1) >= 1)  # at least one jet 
        a = ak.mask(a, mask) # must be done in sequence, this perserves the shape
        mask = (a['JetsAK8.fCoordinates.fPt'][:,0] > 500.) # leading>500
        cutflow['ak8jet.pt>500'] = sum(1 for event in mask if event) # count the number of true entries

        # Triggers
        trigger_indices = np.array([copy.trigger_branch.index(t) for t in copy.triggers])
        if len(a):
            trigger_decisions = a['TriggerPass'].to_numpy()[:,trigger_indices]
            mask = mask & (trigger_decisions == 1).any(axis=-1)
        cutflow['triggers'] = sum(1 for event in mask if event) # count the number of true entries

    # At least 2 AK15 jets
    mask = mask & (ak.count(a['JetsAK15.fCoordinates.fPt'], axis=-1) >= 2)
    cutflow['n_ak15jets>=2'] = sum(1 for event in mask if event) # count the number of true entries

    #JetsAK15_JetID criteria for tight selection cuts: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVRun2018
    #jets_id event level? -->apply it only to sub-leading jet?
    a = ak.mask(a, mask) #apply mask and preserve shape
    mask = mask & (a['JetsAK15_ID'][:,1]>0.)
    cutflow['jetsak15_id'] = sum(1 for event in mask if event) # count the number of true entries

    # At least 2 AK4 jets --> deadcells study
    mask = mask & (ak.count(a['Jets.fCoordinates.fPt'], axis=-1) >= 2)
    cutflow['n_ak4jets>=2'] = sum(1 for event in mask if event) # count the number of true entries

    # subleading eta < 2.4 eta
    a = ak.mask(a, mask) #apply mask and preserve shape
    mask = mask & (np.abs(a['JetsAK15.fCoordinates.fEta'][:,1])<2.4)
    cutflow['subl_eta<2.4'] = sum(1 for event in mask if event) # count the number of true entries

    # positive ECF values
    for ecf in [
        'JetsAK15_ecfC2b1', 'JetsAK15_ecfD2b1',
        'JetsAK15_ecfM2b1', 'JetsAK15_ecfN2b2',
        ]:
        mask = mask & (a[ecf][:,1]>0.)
    cutflow['subl_ecf>0'] = sum(1 for event in mask if event) # count the number of true entries

    # rtx>1.1
    rtx = np.sqrt(1. + a['MET'].to_numpy() / a['JetsAK15.fCoordinates.fPt'][:,1].to_numpy())
    mask = mask & (rtx>1.1)
    cutflow['rtx>1.1'] = sum(1 for event in mask if event) # count the number of true entries


    # muon pt < 1500 filter to avoid highMET events
    mask = mask & (~ak.any(a['Muons.fCoordinates.fPt'] > 1500., axis=-1))
    cutflow['muonpt<1500'] = sum(1 for event in mask if event) # count the number of true entries

    if single_muon_cr:
        # apply preselection - muon veto + muon selection
        # (used medium ID + pt > 50 GeV + iso < 0.2 in EXO-19-020,
        #  see AN-19-061 section 4.2)
        # require the selected muon to match with the HLT muon object
        # (which should be saved in the SingleMuon ntuples) by ΔR < 0.2
        mask = mask & (a['NMuons']>=1)
        if len(a):
            mask = mask & (
                (a['Muons_mediumID'][:,0])
                & (a['Muons.fCoordinates.fPt'][:,0]>50.)
                & (a['Muons_iso'][:,0]<.2) 
            )
        if len(a):
            mask = mask & (ak.count(a['HLTMuonObjects.fCoordinates.fPt'], axis=-1) >= 1)
            mask = mask & (calc_dr(
                a['Muons.fCoordinates.fEta'][:,0].to_numpy(),
                a['Muons.fCoordinates.fPhi'][:,0].to_numpy(),
                a['HLTMuonObjects.fCoordinates.fEta'][:,0].to_numpy(),
                a['HLTMuonObjects.fCoordinates.fPhi'][:,0].to_numpy(),
                ) < .2)
        cutflow['singlemuon'] = sum(1 for event in mask if event) # count the number of true entries
        mask = mask & (a['NElectrons']==0)
        cutflow['nelectrons=0'] = sum(1 for event in mask if event) # count the number of true entries
    else:
        # lepton vetoes
        mask = mask & (a['NMuons']==0) & (a['NElectrons']==0)
        cutflow['nleptons=0'] = sum(1 for event in mask if event) # count the number of true entries

    # MET filters
    for b in [
        'HBHENoiseFilter',
        'HBHEIsoNoiseFilter',
        'eeBadScFilter',
        'ecalBadCalibFilter',
        'BadPFMuonFilter',
        'BadChargedCandidateFilter',
        'globalSuperTightHalo2016Filter',
        ]:
        mask = mask & (a[b]!=0) # Pass events if not 0, is that correct?
    cutflow['metfilter'] = sum(1 for event in mask if event) # count the number of true entries

    # Filter out jets that are too close to dead cells
    ak4jet_eta = a['Jets.fCoordinates.fEta'][:,1].to_numpy()
    ak4jet_phi = a['Jets.fCoordinates.fPhi'][:,1].to_numpy()
    dead_cell_mask = svj.veto_phi_spike(
        svj.dataqcd_eta_ecaldead[array.year], svj.dataqcd_phi_ecaldead[array.year],
        ak4jet_eta, ak4jet_phi,
        rad = 0.01
        )

    mask = mask & dead_cell_mask
    cutflow['ecaldeadcells'] = sum(1 for event in mask if event) # count the number of true entries

    # abs(metdphi)<1.5
    METDphi = svj.calc_dphi(a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy(), a['METPhi'].to_numpy())
    mask = mask & (abs(METDphi)<1.5)
    cutflow['abs(metdphi)<1.5'] = sum(1 for event in mask if event) # count the number of true entries

    cutflow['preselection'] = sum(1 for event in mask if event) # count the number of true entries

    copy.array = a
    logger.debug('cutflow:\n%s', pprint.pformat(copy.cutflow))
    # return copy, mask # magri e' utile di avere il cutflow

    print("made mask")
    mask = ak.to_numpy(mask, allow_missing=True)
    print(sum(1 for event in mask if event)) # count the number of true entries
    return mask 

def make_train_val_test(length, train_frac, val_frac, test_frac, random_seed=None):
    """Make a mask for splitting the data into train, validation, and test data sets

    Args:
        length (int): length of the number of events to be split
        train_frac (float): fraction of the events to go into the training dataset
        val_frac (float): fraction of the events to go into the validation dataset
        test_frac (float): fraction of the events to go into the test dataset
    
    Returns:
        numpy.ndarray: Array containing values 0, 1, and 2 indicating the dataset for each event
    """
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum up to 1.0"

    train_size = int(train_frac * length)
    val_size = int(val_frac * length)
    test_size = int(test_frac * length)

    # create an array of dataset labels
    dataset = np.zeros(length, dtype=int)
    dataset[train_size:train_size + val_size] = 1
    dataset[train_size + val_size:train_size + val_size + test_size] = 2

    # Shuffle the array to randomize the ordering
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(dataset)

    return dataset

#----------------------------------------------------------------------------------------
# Essentially the 'main' function -------------------------------------------------------
#----------------------------------------------------------------------------------------

failed_rootfiles = []

for i, rootfile in enumerate(data.rootfiles):
    try:
        outfile = dst(rootfile)
        if seutils.isfile(outfile):
            svj.logger.info('File %s exists, skipping', outfile)
            continue
        svj.logger.info('Processing rootfile %s/%s: %s', i, len(data.rootfiles)-1, rootfile)

        array = svj.open_root(rootfile)

        # apply preselection and get a mask
        presel_mask = preselection_mask(array)
 
        # make a train, validation, test mask based on splittings by Rob and Cesare
        train_val_test_mask = make_train_val_test(sum(1 for event in presel_mask if event), 0.75, 0.075, 0.175, random_seed=34560)
 
        # Check again, to avoid race conditions
        if seutils.isfile(outfile):
            svj.logger.info('File %s exists now, not staging out', outfile)
            continue
        # Save the mask of events that pass the preselection 
        # make sure the output file has the same name as the input file + mask 
        save_mask(presel_mask, train_val_test_mask, dst)

    except Exception:
        failed_rootfiles.append(rootfile)
        svj.logger.error('Error processing %s; continuing. Error:\n%s', rootfile, traceback.format_exc())


if failed_rootfiles:
    svj.logger.info(
        'Failures were encountered for the following rootfiles:\n%s',
        '\n'.join(failed_rootfiles)
        )

