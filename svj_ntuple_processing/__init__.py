import os, os.path as osp, logging, pprint, uuid
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
import uproot
import awkward as ak


INCLUDE_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "include")
def version():
    with open(osp.join(INCLUDE_DIR, "VERSION"), "r") as f:
        return(f.read().strip())

def uid():
    return str(uuid.uuid4())


def setup_logger(name='svj'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[92m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

UL = True  # Global switch for UL vs PREL


triggers_2018 = [
    # AK8PFJet triggers
    'HLT_AK8PFJet500_v',
    'HLT_AK8PFJet550_v',
    # CaloJet
    'HLT_CaloJet500_NoJetID_v',
    'HLT_CaloJet550_NoJetID_v',
    # PFJet and PFHT
    'HLT_PFHT1050_v',
    'HLT_PFJet500_v',
    'HLT_PFJet550_v',
    # Trim mass jetpt+HT
    'HLT_AK8PFHT800_TrimMass50_v',
    'HLT_AK8PFHT850_TrimMass50_v',
    'HLT_AK8PFHT900_TrimMass50_v',
    'HLT_AK8PFJet400_TrimMass30_v',
    'HLT_AK8PFJet420_TrimMass30_v',
    # MET triggers
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v',
    'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v',
    'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v',
    ]

triggers_2016 = [
    'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v',
    'HLT_AK8PFJet360_TrimMass30_v',
    'HLT_CaloJet500_NoJetID_v',
    'HLT_PFHT900_v',
    'HLT_PFJet450_v',
    'HLT_PFJet500_v',
    ]

# Checked: 2017 identical to 2018
triggers_per_year = {2016: triggers_2016, 2017: triggers_2018, 2018: triggers_2018}


class Arrays:
    """
    Container class for an awkward.Array object + metadata about it.

    The functions filter_preselection, filter_zprime_in_cone, and filter_stitch
    can be used on this object to apply event selections.

    This object is meant to be converted to a Columns object before being
    saved to disk.
    """
    def __init__(self, array=None):
        self.array = array
        self.trigger_branch = None
        self.cutflow = OrderedDict()
        self.metadata = {'year' : 2018}

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return '<Arrays {0}>'.format(pprint.pformat(self.metadata))

    def cut(self, cut_name):
        """Adds an entry to the cutflow list now"""
        self.cutflow[cut_name] = len(self)

    def copy(self):
        copy = Arrays(self.array)
        copy.trigger_branch = self.trigger_branch
        copy.cutflow = self.cutflow.copy()
        copy.metadata = self.metadata.copy()
        return copy

    @property
    def year(self):
        return self.metadata['year']

    @property
    def bkg_type(self):
        if 'bkg_type' in self.metadata:
            return self.metadata['bkg_type']
        # TODO: Something based on filename

    @property
    def triggers(self):
        return triggers_per_year[self.year]


@contextmanager
def local_copy(remote):
    """
    Creates a temporary local copy of a remote file
    """
    import seutils
    must_delete = False
    try:
        if seutils.path.has_protocol(remote):
            # File is remote, make local copy
            must_delete = True
            local = uid() + osp.splitext(remote)[1]
            logger.info('Copying %s -> %s', remote, local)
            seutils.cp(remote, local)
            yield local
        else:
            # File is already local, nothing to do
            yield remote
    finally:
        if must_delete:
            try:
                os.remove(local)
            except Exception:
                pass


def open_root(rootfile):
    """
    Returns an Arrays object from a rootfile (unfiltered).
    """
    branches = [
        'JetsAK8.fCoordinates.fPt',
        'JetsAK15.fCoordinates.fPt', 'JetsAK15.fCoordinates.fEta',
        'JetsAK15.fCoordinates.fPhi', 'JetsAK15.fCoordinates.fE',
        'JetsAK15_ecfC2b1', 'JetsAK15_ecfC2b2',
        'JetsAK15_ecfD2b1', 'JetsAK15_ecfD2b2',
        'JetsAK15_ecfM2b1', 'JetsAK15_ecfM2b2',
        'JetsAK15_ecfN2b1', 'JetsAK15_ecfN2b2',
        'JetsAK15_girth', 'JetsAK15_ptD',
        'JetsAK15_axismajor', 'JetsAK15_axisminor',
        'MET', 'METPhi',
        'TriggerPass',
        'NMuons', 'NElectrons',
        'HBHENoiseFilter', 'HBHEIsoNoiseFilter', 'eeBadScFilter',
        'ecalBadCalibFilter' if UL else 'ecalBadCalibReducedFilter',
        'BadPFMuonFilter', 'BadChargedCandidateFilter', 'globalSuperTightHalo2016Filter',
        'Weight',
        'madHT', 'GenMET', 
        'GenParticles_PdgId',
        'GenParticles_Status',
        'GenParticles.fCoordinates.fPt',
        'GenParticles.fCoordinates.fEta',
        'GenParticles.fCoordinates.fPhi',
        'GenParticles.fCoordinates.fE',
        ]

    with local_copy(rootfile) as local:
        tree = uproot.open(local + ':TreeMaker2/PreSelection')
        arrays = Arrays(tree.arrays(branches))

    # Store the order of trigger names in the array object
    arrays.trigger_branch = tree['TriggerPass'].title.split(',')
    arrays.metadata['src'] = rootfile
    arrays.cut('raw')
    return arrays


def calc_dphi(phi1, phi2):
    """
    Calculates delta phi. Assures output is within -pi .. pi.
    """
    twopi = 2.*np.pi
    # Map to 0..2pi range
    dphi = (phi1 - phi2) % twopi
    # Map pi..2pi --> -pi..0
    dphi[dphi > np.pi] -= twopi
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1-eta2)**2 + calc_dphi(phi1, phi2)**2)


def calculate_mt(pt, eta, phi, e, met, metphi):
    """
    Calculates the transverse mass MT and RT (closely related calcs)
    """
    met_x = np.cos(metphi) * met
    met_y = np.sin(metphi) * met
    jet_x = np.cos(phi) * pt
    jet_y = np.sin(phi) * pt
    # jet_e = np.sqrt(jets.mass2 + jets.pt**2)
    # m^2 + pT^2 = E^2 - pT^2 - pz^2 + pT^2 = E^2 - pz^2
    pz = pt * np.sinh(eta)
    transverse_e = np.sqrt(e**2 - pz**2)
    mt = np.sqrt( (transverse_e + met)**2 - (jet_x + met_x)**2 - (jet_y + met_y)**2 )
    return mt


def filter_preselection(array):
    copy = array.copy()
    a = copy.array
    cutflow = copy.cutflow
    
    # AK8Jet.pT>500
    a = a[ak.count(a['JetsAK8.fCoordinates.fPt'], axis=-1)>=1] # At least one jet
    a = a[a['JetsAK8.fCoordinates.fPt'][:,0]>500.] # leading>500
    cutflow['ak8jet.pt>500'] = len(a)

    # Triggers
    trigger_indices = np.array([copy.trigger_branch.index(t) for t in copy.triggers])
    if len(a):
        trigger_decisions = a['TriggerPass'].to_numpy()[:,trigger_indices]
        a = a[(trigger_decisions == 1).any(axis=-1)]
    cutflow['triggers'] = len(a)

    # At least 2 AK15 jets
    a = a[ak.count(a['JetsAK15.fCoordinates.fPt'], axis=-1) >= 2]
    cutflow['n_ak15jets>=2'] = len(a)

    # subleading eta < 2.4 eta
    a = a[a['JetsAK15.fCoordinates.fEta'][:,1]<2.4]
    cutflow['subl_eta<2.4'] = len(a)

    # positive ECF values
    for ecf in [
        'JetsAK15_ecfC2b1', 'JetsAK15_ecfD2b1',
        'JetsAK15_ecfM2b1', 'JetsAK15_ecfN2b2',
        ]:
        a = a[a[ecf][:,1]>0.]
    cutflow['subl_ecf>0'] = len(a)

    # rtx>1.1
    rtx = np.sqrt(1. + a['MET'].to_numpy() / a['JetsAK15.fCoordinates.fPt'][:,1].to_numpy())
    a = a[rtx>1.1]
    cutflow['rtx>1.1'] = len(a)

    # lepton vetoes
    a = a[(a['NMuons']==0) & (a['NElectrons']==0)]
    cutflow['nleptons=0'] = len(a)

    # MET filters
    for b in [
        'HBHENoiseFilter',
        'HBHEIsoNoiseFilter',
        'eeBadScFilter',
        'ecalBadCalibFilter' if UL else 'ecalBadCalibReducedFilter',
        'BadPFMuonFilter',
        'BadChargedCandidateFilter',
        'globalSuperTightHalo2016Filter',
        ]:
        a = a[a[b]!=0] # Pass events if not 0, is that correct?
    cutflow['metfilter'] = len(a)
    cutflow['preselection'] = len(a)

    copy.array = a
    logger.debug('cutflow:\n%s', pprint.pformat(copy.cutflow))
    return copy


def filter_zprime_in_cone(array):
    copy = array.copy()
    a, cutflow = copy.array, copy.cutflow

    # at least 1 zprime particle
    a = a[ak.sum(a['GenParticles_PdgId']==4900023, axis=-1)>=1]
    # at least 2 dark quarks
    a = a[ak.sum((np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71), axis=-1)>=2]
    cutflow['1zprime2darkquarks'] = len(a)

    # Require both dark quarks and the zprime to be within 1.5 cone of the subleading jet
    eta_subl = a['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    phi_subl = a['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    eta_zprime = a['GenParticles.fCoordinates.fEta'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    phi_zprime = a['GenParticles.fCoordinates.fPhi'][a['GenParticles_PdgId']==4900023][:,0].to_numpy()
    select_darkquarks = (np.abs(a['GenParticles_PdgId'])==4900101) & (a['GenParticles_Status']==71)
    eta_darkquark1 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,0].to_numpy()
    phi_darkquark1 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,0].to_numpy()
    eta_darkquark2 = a['GenParticles.fCoordinates.fEta'][select_darkquarks][:,1].to_numpy()
    phi_darkquark2 = a['GenParticles.fCoordinates.fPhi'][select_darkquarks][:,1].to_numpy()
    a = a[
        (calc_dr(eta_subl, phi_subl, eta_zprime, phi_zprime)<=1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark1, phi_darkquark1)<=1.5)
        & (calc_dr(eta_subl, phi_subl, eta_darkquark2, phi_darkquark2)<=1.5)
        ]
    cutflow['zdq<1.5'] = len(a)
    copy.array = a
    return copy


def filter_stitch(array):
    """
    Filter to cut away kinematic regions from ttjets/wjets that are covered in
    dedicated samples
    """
    copy = array.copy()
    a = copy.array

    if array.bkg_type == 'ttjets':
        if 'htbin' in array.metadata:
            # HT bin
            a = a[a['madHT']>=600.]
        elif 'n_lepton_sample' in array.metadata:
            # SingleLep or DiLep
            if array.metadata.get('genmet_sample', False):
                a = a[(a['madHT']<600.) & (a['GenMET']>=150.)]
            else:
                a = a[(a['madHT']<600.) & (a['GenMET']<150.)]
        else:
            # Inclusive
            genparticle_pdgid = np.abs(a['GenParticles_PdgId'])
            n_leptons = (
                ak.sum(genparticle_pdgid==11, axis=-1)
                + ak.sum(genparticle_pdgid==13, axis=-1)
                + ak.sum(genparticle_pdgid==15, axis=-1)
                )
            a = a[(a['madHT']<600.) & (n_leptons==0)]

    elif array.bkg_type == 'wjets':
        if 'htbin' in array.metadata:
            a = a[a['madHT']>100.]
        else:
            # Inclusive
            a = a[a['madHT']<=100.]

    copy.array = a
    copy.cut('stitch')
    return copy


class Columns:
    """
    Class that contains a dictionary of arrays. Each array is guaranteed
    to be shaped (n_events,).

    This class is designed for read/write to disk.
    """
    @classmethod
    def load(cls, infile, encoding='ASCII'):
        try:
            with local_copy(infile) as local:
                d = np.load(local, allow_pickle=True, encoding=encoding)
        except:
            logger.error('Error opening %s', infile)
            raise
        inst = cls()
        inst.arrays = d['arrays'].item()
        inst.metadata = d['metadata'].item()
        inst.metadata['src'] = infile
        for key, val in zip(d['cutflow_keys'], d['cutflow_vals']):
            inst.cutflow[key] = val
        return inst

    def __init__(self):
        self.arrays = {}
        self.metadata = {}
        self.cutflow = OrderedDict()

    def __len__(self):
        for v in self.arrays.values():
            return len(v)

    def __repr__(self):
        return '<Columns {0}>'.format(pprint.pformat(self.metadata))

    def save(self, outfile):
        import seutils
        do_stageout = False
        if seutils.path.has_protocol(outfile):
            remote_outfile = outfile
            outfile = uid() + '.npz'
            do_stageout = True

        cutflow_keys = []
        cutflow_vals = []
        for key, val in self.cutflow.items():
            cutflow_keys.append(key)
            cutflow_vals.append(val)
        cutflow_vals = np.array(cutflow_vals)

        logger.info('Dumping to %s', outfile)

        # Automatically create parent directory if not existent
        outdir = osp.dirname(osp.abspath(outfile))
        if not osp.isdir(outdir):
            os.makedirs(outdir)

        np.savez(
            outfile,
            arrays = self.arrays,
            metadata = dict(self.metadata, svj_ntuple_processing_version=version()),
            cutflow_keys = cutflow_keys,
            cutflow_vals = cutflow_vals
            )

        if do_stageout:
            logger.info('Staging out %s -> %s', outfile, remote_outfile)
            seutils.cp(outfile, remote_outfile)
            os.remove(outfile)

    def to_numpy(self, features=None):
        """
        Returns the various arrays as a rectangular numpy array.
        If `features` is None, all features are used, sorted alphabetically.
        """
        if features is None: features = list(sorted(self.arrays.keys()))
        # Check if passed features exist in this .npz
        missing_features = []
        for f in features:
            if f not in self.arrays.keys():
                logger.error(f'Feature {f} is not available in {self}.')
                missing_features.append(f)
        if missing_features:
            raise Exception(
                f'Cannot build numpy array.'
                f' Available features: {list(self.arrays.keys())};'
                f' Missing requested features: {missing_features}'
                )
        X = []
        for f in features:
            X.append(self.arrays[f])
        X = np.column_stack(X)
        return X

    def to_dataframe(self, features=None):
        """
        Returns the various arrays as a pandas.DataFrame. The self.arrays keys are used
        as column names
        """
        import pandas as pd
        return pd.DataFrame.from_dict(self.arrays)

    def copy(self):
        copy = self.__class__()
        copy.metadata = self.metadata.copy()
        copy.arrays = self.arrays.copy()
        copy.cutflow = self.cutflow.copy()
        return copy


def load_numpy(infile, features):
    """
    Convenience function
    """
    return Columns.load(infile).to_numpy(features)


def concat_columns(columns):
    """
    Concatenates columns into one column object.
    Metadata is taken from the first column.
    Keys are taken from first column, and are assumed to exist in others!
    """
    cols = Columns()
    cols.metadata = columns[0].metadata.copy()

    # Concatenated arrays; 1 call per key
    for key in columns[0].arrays.keys():
        try:
            cols.arrays[key] = np.concatenate([c.arrays[key] for c in columns])
        except KeyError:
            # Find the column that crashed:
            for c in columns:
                if key not in c.arrays:
                    logger.error(
                        'Key %s does not exist in columns %s;'
                        ' expected columns for concatenation: %s;'
                        ' available columns: {list(c.arrays.keys())}.',
                        key, c, list(columns[0].arrays.keys())
                        )
            raise

    # Summed cutflow
    for key in columns[0].cutflow.keys():
        cols.cutflow[key] = sum(c.cutflow[key] for c in columns)

    return cols


def bdt_feature_columns(array):
    """
    Takes an Array object, calculates needed columns for the bdt training.
    """
    cols = Columns()
    cols.metadata = array.metadata.copy()
    cols.cutflow = array.cutflow.copy()
    # Prepare features
    arr = array.array
    a = {}
    a['girth'] = arr['JetsAK15_girth'][:,1].to_numpy()
    a['ptd'] = arr['JetsAK15_ptD'][:,1].to_numpy()
    a['axismajor'] = arr['JetsAK15_axismajor'][:,1].to_numpy()
    a['axisminor'] = arr['JetsAK15_axisminor'][:,1].to_numpy()
    a['ecfm2b1'] = arr['JetsAK15_ecfM2b1'][:,1].to_numpy()
    a['ecfd2b1'] = arr['JetsAK15_ecfD2b1'][:,1].to_numpy()
    a['ecfc2b1'] = arr['JetsAK15_ecfC2b1'][:,1].to_numpy()
    a['ecfn2b2'] = arr['JetsAK15_ecfN2b2'][:,1].to_numpy()
    a['metdphi'] = calc_dphi(arr['JetsAK15.fCoordinates.fEta'][:,1].to_numpy(), arr['METPhi'].to_numpy())

    a['weight'] = arr['Weight'].to_numpy()
    a['met'] = arr['MET'].to_numpy()
    a['metphi'] = arr['METPhi'].to_numpy()

    # Save some extra vars for potential reweighting / other analysis
    a['pt'] = arr['JetsAK15.fCoordinates.fPt'][:,1].to_numpy()
    a['eta'] = arr['JetsAK15.fCoordinates.fEta'][:,1].to_numpy()
    a['phi'] = arr['JetsAK15.fCoordinates.fPhi'][:,1].to_numpy()
    a['e'] = arr['JetsAK15.fCoordinates.fE'][:,1].to_numpy()
    a['mt'] = calculate_mt(
        a['pt'], a['eta'], a['phi'], a['e'],
        a['met'], a['metphi']
        )
    a['rt'] = np.sqrt(1.+a['met']/a['pt'])

    a['leading_pt'] = arr['JetsAK15.fCoordinates.fPt'][:,0].to_numpy()
    a['leading_eta'] = arr['JetsAK15.fCoordinates.fEta'][:,0].to_numpy()
    a['leading_phi'] = arr['JetsAK15.fCoordinates.fPhi'][:,0].to_numpy()
    a['leading_e'] = arr['JetsAK15.fCoordinates.fE'][:,0].to_numpy()

    cols.arrays = a
    return cols
