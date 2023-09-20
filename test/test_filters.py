import pytest
import os, os.path as osp
import numpy as np
import svj_ntuple_processing as svj
from collections import OrderedDict

TESTDIR = osp.dirname(osp.abspath(__file__))
if not TESTDIR.endswith('/'): TESTDIR += '/'


def test_preselection_local():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    len_a_prefilter = len(a)
    a = svj.filter_preselection(a)
    assert 0.05*len_a_prefilter < len(a) < .5*len_a_prefilter


def test_filter_zprime_in_cone():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    a = svj.filter_preselection(a)
    len_a_prefilter = len(a)
    a = svj.filter_zprime_in_cone(a)
    assert 0.1*len_a_prefilter < len(a) < .8*len_a_prefilter


def test_preselection_remote():
    a = svj.open_root('root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.root')
    len_a_prefilter = len(a)
    a = svj.filter_preselection(a)
    assert 0.05*len_a_prefilter < len(a) < .5*len_a_prefilter


def test_stitch_ttjets():
    a = svj.open_root(TESTDIR + 'ttjets_inclusive.root')

    a.metadata['bkg_type'] = 'ttjets'

    a_filtered = svj.filter_stitch(a)
    assert len(a_filtered) < len(a)

    a.metadata['htbin'] = [600,800] # Pretend it's an HT-600to800 sample
    a_filtered = svj.filter_stitch(a)
    assert len(a_filtered) < len(a)

    del a.metadata['htbin']
    a.metadata['n_lepton_sample'] = 1 # Pretend it's a SingleLep sample
    a_filtered = svj.filter_stitch(a)
    assert len(a_filtered) < len(a)


def test_bdt_feature_columns():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    a = svj.filter_preselection(a)
    a = svj.filter_zprime_in_cone(a)
    cols = svj.bdt_feature_columns(a)

    assert 'girth' in cols.arrays

    # Assert all shapes equal
    shape = None
    for key, val in cols.arrays.items():
        if shape is None:
            shape = val.shape
        else:
            assert val.shape == shape


def test_columns_io_simple():
    cols = svj.Columns()
    cols.cutflow['cut1'] = 200
    cols.cutflow['cut2'] = 100
    cols.metadata['testkey'] = 'testval'
    cols.metadata['testnum'] = 123
    cols.arrays = {'arr1' : np.ones(4), 'arr2' : np.zeros(4)}

    cols.save('testcols.npz')
    cols2 = svj.Columns.load('testcols.npz')

    assert cols.cutflow == cols2.cutflow
    # Remove src before comparison; those are expected to be different
    cols.metadata.pop('src', None)
    cols2.metadata.pop('src', None)
    cols2.metadata.pop('svj_ntuple_processing_version', None)
    assert cols.metadata == cols2.metadata
    assert set(cols.arrays.keys()) == set(cols2.arrays.keys())
    for k in cols.arrays.keys():
        np.testing.assert_array_equal(cols.arrays[k], cols2.arrays[k])


def test_columns_io_practice():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    a = svj.filter_preselection(a)
    a = svj.filter_zprime_in_cone(a)
    cols = svj.bdt_feature_columns(a)

    cols.save('testcols.npz')
    cols2 = svj.Columns.load('testcols.npz')

    assert cols.cutflow == cols2.cutflow
    # Remove src before comparison; those are expected to be different
    cols.metadata.pop('src', None)
    cols2.metadata.pop('src', None)
    cols2.metadata.pop('svj_ntuple_processing_version', None)
    assert cols.metadata == cols2.metadata
    assert set(cols.arrays.keys()) == set(cols2.arrays.keys())
    for k in cols.arrays.keys():
        np.testing.assert_array_equal(cols.arrays[k], cols2.arrays[k])


def test_columns_io_remote():
    import seutils
    outfile = 'root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/test.npz'
    if seutils.isfile(outfile): seutils.rm(outfile)

    cols = svj.Columns()
    cols.cutflow['cut1'] = 200
    cols.cutflow['cut2'] = 100
    cols.metadata['testkey'] = 'testval'
    cols.metadata['testnum'] = 123
    cols.arrays = {'arr1' : np.ones(4), 'arr2' : np.zeros(4)}

    cols.save(outfile)
    cols2 = svj.Columns.load(outfile)

    assert cols.cutflow == cols2.cutflow
    # Remove src before comparison; those are expected to be different
    cols.metadata.pop('src', None)
    cols2.metadata.pop('src', None)
    cols2.metadata.pop('svj_ntuple_processing_version', None)
    assert cols.metadata == cols2.metadata
    assert set(cols.arrays.keys()) == set(cols2.arrays.keys())
    for k in cols.arrays.keys():
        np.testing.assert_array_equal(cols.arrays[k], cols2.arrays[k])


def test_concat_columns():
    col1 = svj.Columns()
    col1.cutflow['cut1'] = 200
    col1.cutflow['cut2'] = 100
    col1.metadata['testkey'] = 'testval'
    col1.metadata['testnum'] = 123
    col1.arrays = {'arr1' : np.ones(4), 'arr2' : np.zeros(4)}

    col2 = svj.Columns()
    col2.cutflow['cut1'] = 20
    col2.cutflow['cut2'] = 10
    col2.metadata['testkey'] = 'testval2'
    col2.metadata['testnum'] = 1234
    col2.arrays = {'arr1' : np.ones(4), 'arr2' : np.zeros(4)}

    col = svj.concat_columns((col1, col2))
    assert col.cutflow == OrderedDict(cut1=220, cut2=110)
    np.testing.assert_array_equal(col.arrays['arr1'], np.ones(8))
    np.testing.assert_array_equal(col.arrays['arr2'], np.zeros(8))


def test_load_numpy():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    a = svj.filter_preselection(a)
    a = svj.filter_zprime_in_cone(a)
    cols = svj.bdt_feature_columns(a)
    cols.save('testcols.npz')

    X = svj.load_numpy('testcols.npz', ['girth', 'ptd', 'axismajor', 'axisminor'])
    assert X.shape == (len(cols), 4)


def test_trigger_column():
    a = svj.open_root(TESTDIR + 'madpt300_mz350_mdark10_rinv0.3.root')
    a = svj.filter_at_least_one_ak8jet(a)
    col = svj.triggerstudy_columns(a, single_muon_trigs=True)
    assert col.arrays['mt_ak15_subl'].shape == col.arrays['pt'].shape
    assert col.arrays['mt_ak15_subl'].shape == col.arrays['pt_ak15_subl'].shape
    assert col.arrays['pt_ak15'].shape == col.arrays['pt_ak15_subl'].shape
    assert np.any(np.isfinite(col.arrays['mt_ak15_subl']))
    assert 'HLT_Mu50_v' in col.metadata['trigger_titles']


def test_singlemuon_cr():
    f = TESTDIR + 'data2018_singlemuon.root'
    # f = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV20/Run2018A-UL2018-v3/SingleMuon/954_RA2AnalysisTree.root'
    a = svj.open_root(f, load_gen=False, load_hlt=True)
    a = svj.filter_preselection(a, single_muon_cr=True)
    assert 'singlemuon' in a.cutflow
    assert 'triggers' not in a.cutflow
    assert 'ak8jet.pt>500' not in a.cutflow
    col = svj.triggerstudy_columns(a, is_mc=False, all_triggers=True)
    assert 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v' in col.metadata['trigger_titles']
    # raise 0