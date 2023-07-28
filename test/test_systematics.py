import os, os.path as osp
import numpy as np
import awkward as ak
import svj_ntuple_processing as svj

TESTDIR = osp.dirname(osp.abspath(__file__))
if not TESTDIR.endswith('/'): TESTDIR += '/'


def assert_akarray_almost_equal(a, b, tol=0.001):
    eq = np.abs(a - b) < tol
    if not ak.all(eq):
        i_faulty = np.nonzero(ak.any(~eq, axis=-1).to_numpy())[0]
        i = i_faulty[0]
        raise AssertionError(
            f'Not equal at idx {i_faulty}, e.g.:'
            f'\n  a[{i}] = {a[i]}, b[{i}] = {b[i]}'
            )


def test_reverse_indexing():
    # for(unsigned j = 0; j < Jets_origIndex.size(); ++j){
    #     //reverse the index vector
    #     newIndex[Jets_origIndex[j]] = j;
    # }

    orig_index = ak.Array([[1, 2, 0], [4, 1, 3, 0, 2]])
    ans = ak.Array([[2, 0, 1], [3, 1, 4, 2, 0]])

    manual_loop = []
    for i_event in range(len(orig_index)):
        n = len(orig_index[i_event])
        tmp = [-1] * n
        for j in range(n):
            tmp[orig_index[i_event,j]] = j
        manual_loop.append(tmp)
    manual_loop = ak.Array(manual_loop)

    argsort = ak.argsort(orig_index, axis=-1)

    print(f'ans={ans}')
    print(f'man={manual_loop}')
    print(f'arg={argsort}')

    assert ak.all(ans == manual_loop)
    assert ak.all(ans == argsort)


def test_index_associativity():
    np.random.seed(1001)
    arr = np.arange(20)

    order1 = np.arange(20)
    np.random.shuffle(order1)
    order2 = np.arange(20)
    np.random.shuffle(order2)

    ans = arr[order1][order2]
    right_first = arr[order1[order2]]

    print(f'ans={ans}')
    print(f'right_first={right_first}')
    np.testing.assert_array_equal(ans, right_first)


def test_multiplying_ptetaphi_equals_pxpypz():
    """
    Test multiplying a 4-vector by a constant

    In ROOT, TLorentzVector *= (double)c means doing fP*=c and fE*=c:
    https://root.cern.ch/doc/master/TLorentzVector_8h_source.html#l00399
    Where fP is a 3-vector (simply element-wise *c) and fE is the energy.

    This test aims to ascertain that simply multiplying pT and E by c, and leaving
    eta and phi unchanged, is the same thing as multiplying px, py, pz, and E
    independently by c.
    """
    # Create two jets
    pt = np.array([300., 200.])
    eta = np.array([.5, 1.2])
    phi = np.array([.5*np.pi, 1.3*np.pi])
    e = np.array([250., 190.])

    # The correction factor
    corr = 1.1

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    # Apply correction
    px *= corr
    py *= corr
    pz *= corr

    # Calculate corrected pT and eta
    pt_corr = np.sqrt(px**2 + py**2)
    eta_corr = np.arcsinh(pz / pt_corr)

    # Ensure pt_corr == pt * corr, and eta_corr == eta
    np.testing.assert_array_almost_equal(pt*corr, pt_corr)
    np.testing.assert_array_almost_equal(eta, eta_corr)



def test_syst():
    rootfile = osp.join(TESTDIR, 'madpt300_mz350_mdark10_rinv0.3_ak15jecjer.root')
    arrays = svj.open_root(rootfile, load_gen=True, load_jerjec=True)

    pt = arrays.array['JetsAK15.fCoordinates.fPt']
    eta = arrays.array['JetsAK15.fCoordinates.fEta']
    phi = arrays.array['JetsAK15.fCoordinates.fPhi']
    energy = arrays.array['JetsAK15.fCoordinates.fE']
    print(pt)

    jer_factor = arrays.array['JetsAK15_jerFactor']
    jec_unc = arrays.array['JetsAK15_jecUnc']
    map_final_idx_to_orig_idx = arrays.array['JetsAK15_origIndex']

    njets = ak.count(arrays.array['JetsAK15_origIndex'], axis=-1)

    # Final JetsAK15 is ordered by pT _after_ corrections
    # [[1, 2, 0], ...] -> Jet 0 was originally jet 1 before corrections
    # Get reverse map: original jet 1 is now at 0
    # [[2, 0, 1], ...]
    # --> argsort([1, 2, 0]) = [2, 0, 1]
    # --> argsort([0, 2, 1]) = [0, 2, 1], for single flip argsort does not change anything
    map_orig_idx_to_final_idx = ak.argsort(map_final_idx_to_orig_idx, axis=-1)


    JetsAK15JECup_origIndex = arrays.array['JetsAK15JECup_origIndex']
    JetsAK15JECup_jerFactor = arrays.array['JetsAK15JECup_jerFactor']

    manual = []
    for i_event in range(len(njets)):
        tmp = []
        for j in range(njets[i_event]):
            i = map_orig_idx_to_final_idx[i_event][JetsAK15JECup_origIndex[i_event][j]]
            up = pt[i_event][i] / jer_factor[i_event][i] * (1.+jec_unc[i_event][i]) * JetsAK15JECup_jerFactor[i_event][j]
            tmp.append(up)
        manual.append(tmp)

    manual = ak.Array(manual)
    print(manual)

    # ak solution
    map_unc_idx_to_orig_idx = JetsAK15JECup_origIndex
    map_orig_idx_to_unc_idx = ak.argsort(map_unc_idx_to_orig_idx, axis=-1)

    reorder_final_to_unc = map_final_idx_to_orig_idx[map_orig_idx_to_unc_idx]

    pt = pt[reorder_final_to_unc]
    jer_factor = jer_factor[reorder_final_to_unc]
    jec_unc = jec_unc[reorder_final_to_unc]

    ak_ans = pt / jer_factor * (1.+jec_unc) * JetsAK15JECup_jerFactor
    print(ak_ans)

    assert_akarray_almost_equal(manual, ak_ans)


    # If done correctly, ak_ans should be sorted (descending)
    # Sort by pt, and check if nothing changed:
    pt_sorted = ak.sort(ak_ans, axis=-1, ascending=False)
    assert_akarray_almost_equal(ak_ans, pt_sorted)

    # Now test the algorithm
    pt_corr, eta_corr, phi_corr, energy_corr = svj.calc_jec_variation(
        arrays.array['JetsAK15.fCoordinates.fPt'],
        arrays.array['JetsAK15.fCoordinates.fEta'],
        arrays.array['JetsAK15.fCoordinates.fPhi'],
        arrays.array['JetsAK15.fCoordinates.fE'],
        jer_factor = arrays.array['JetsAK15_jerFactor'],
        jec_unc = arrays.array['JetsAK15_jecUnc'],
        orig_idx = arrays.array['JetsAK15_origIndex'],
        variation_orig_idx = arrays.array['JetsAK15JECup_origIndex'],
        variation_jer_factor = arrays.array['JetsAK15JECup_jerFactor'],
        )
    
    print(pt_corr)
    assert_akarray_almost_equal(pt_corr, ak_ans)


    # From Alex:
    # JetsJECupFriend[j] = Jets->at(i)*(1./Jets_jerFactor->at(i))*(1+Jets_jecUnc->at(i))*JetsJECup_jerFactor->at(j);

    # From Kevin:
    # vector<int> newIndex(Jets_origIndex.size(),-1);
    # for(unsigned j = 0; j < Jets_origIndex.size(); ++j){
    #     //reverse the index vector
    #     newIndex[Jets_origIndex[j]] = j;
    # }
    # vector<TLorentzVector> JetsJECup(Jets.size());
    # for(unsigned j = 0; j < JetsJECup_origIndex.size(); ++j){
    #     //JetsJECup_origIndex is sorted in the final order after JEC uncertainty variation
    #     //go up to common ancestor, then down to central smeared collection
    #     int i = newIndex[JetsJECup_origIndex[j]];
    #     //undo central smearing, apply JEC unc, redo smearing w/ new smearing factor
    #     JetsJECup[j] = Jets[i]/Jets_jerFactor[i]*(1+Jets_jecUnc[i])*JetsJECup_jerFactor[j];
    # }

    
    # 'JetsAK15JECdown_jerFactor'
    # 'JetsAK15JECdown_origIndex'
    # 'JetsAK15JECup_jerFactor'
    # 'JetsAK15JECup_origIndex'
    # 'JetsAK15JERdown_origIndex'
    # 'JetsAK15JERup_origIndex'
    # 'JetsAK15_origIndex'
    # 'JetsAK15_jecUnc'
    # 'JetsAK15_jerFactor'
    # 'JetsAK15_jerFactorDown'
    # 'JetsAK15_jerFactorUp'


def test_jer_up():
    rootfile = osp.join(TESTDIR, 'madpt300_mz350_mdark10_rinv0.3_ak15jecjer.root')
    arrays = svj.open_root(rootfile, load_gen=True, load_jerjec=True)
    pt_corr, eta_corr, phi_corr, energy_corr = svj.calc_jer_variation(
        arrays.array['JetsAK15.fCoordinates.fPt'],
        arrays.array['JetsAK15.fCoordinates.fEta'],
        arrays.array['JetsAK15.fCoordinates.fPhi'],
        arrays.array['JetsAK15.fCoordinates.fE'],
        jer_factor = arrays.array['JetsAK15_jerFactor'],
        orig_idx = arrays.array['JetsAK15_origIndex'],
        variation_orig_idx = arrays.array['JetsAK15JERup_origIndex'],
        variation_jer_factor = arrays.array['JetsAK15_jerFactorUp'],
        )
    # no assert, just check if it doesn't crash
    print('All good')


def test_application():
    rootfile = osp.join(TESTDIR, 'madpt300_mz350_mdark10_rinv0.3_ak15jecjer.root')
    arrays = svj.open_root(rootfile, load_gen=True, load_jerjec=True)

    variation = svj.apply_jer_up(arrays)

    # Eta should at most be resorted, but no different values
    assert_akarray_almost_equal(
        ak.sort(arrays.array['Jets.fCoordinates.fEta'], axis=-1),
        ak.sort(variation.array['Jets.fCoordinates.fEta'], axis=-1)
        )
    assert ak.any(arrays.array['Jets.fCoordinates.fPt'] != variation.array['Jets.fCoordinates.fPt'])
    assert ak.any(arrays.array['MET'] != variation.array['MET'])

    cols = svj.bdt_feature_columns(variation)

if __name__ == '__main__':
    rootfile = osp.join(TESTDIR, 'madpt300_mz350_mdark10_rinv0.3_ak15jecjer.root')
    arrays = svj.open_root(rootfile, load_gen=True, load_jerjec=True)