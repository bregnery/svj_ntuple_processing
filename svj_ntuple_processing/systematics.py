import svj_ntuple_processing as svj

import awkward as ak
import numpy as np


def calc_jec_variation(
        pt, eta, phi, energy,
        jer_factor, jec_unc, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JEC variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        jec_unc (ak.Array): the JEC uncertainty
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, and energy after
            applying the variation and reordered by the pT after variation.
    """

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]
    jec_unc = jec_unc[reorder_final_to_var]

    corr = 1. / jer_factor * (1.+jec_unc) * variation_jer_factor
    return pt*corr, eta, phi, energy*corr


def calc_jer_variation(
        pt, eta, phi, energy,
        jer_factor, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JER variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, and energy after
            applying the variation and reordered by the pT after variation.
    """
	# for(unsigned j = 0; j < JetsJERup_origIndex->size(); ++j){
    # 	int i = newIndex[JetsJERup_origIndex->at(j)];
    # 	JetsJERupFriend[j] = Jets->at(i)*(1./Jets_jerFactor->at(i))*Jets_jerFactorUp->at(i);
	# }

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]

    corr = 1. / jer_factor * variation_jer_factor
    return pt*corr, eta, phi, energy*corr


def apply_jer_up(arrays):
    """
    Macro to apply the JER up variation on Arrays object.
    """
    arrays = arrays.copy()
    for jets in ['Jets', 'JetsAK8', 'JetsAK15']:
        pt, eta, phi, energy = calc_jer_variation(
            arrays.array[jets+'.fCoordinates.fPt'],
            arrays.array[jets+'.fCoordinates.fEta'],
            arrays.array[jets+'.fCoordinates.fPhi'],
            arrays.array[jets+'.fCoordinates.fE'],
            jer_factor = arrays.array[jets+'_jerFactor'],
            orig_idx = arrays.array[jets+'_origIndex'],
            variation_orig_idx = arrays.array[jets+'JERup_origIndex'],
            variation_jer_factor = arrays.array[jets+'_jerFactorUp'],
            )
        arrays.array[jets+'.fCoordinates.fPt'] = pt
        arrays.array[jets+'.fCoordinates.fEta'] = eta
        arrays.array[jets+'.fCoordinates.fPhi'] = phi
        arrays.array[jets+'.fCoordinates.fE'] = energy

    arrays.array['MET'] = arrays.array['METUp'][:,0]
    return arrays


def apply_jer_down(arrays):
    """
    Macro to apply the JER down variation on Arrays object.
    """
    arrays = arrays.copy()
    for jets in ['Jets', 'JetsAK8', 'JetsAK15']:
        pt, eta, phi, energy = calc_jer_variation(
            arrays.array[jets+'.fCoordinates.fPt'],
            arrays.array[jets+'.fCoordinates.fEta'],
            arrays.array[jets+'.fCoordinates.fPhi'],
            arrays.array[jets+'.fCoordinates.fE'],
            jer_factor = arrays.array[jets+'_jerFactor'],
            orig_idx = arrays.array[jets+'_origIndex'],
            variation_orig_idx = arrays.array[jets+'JERdown_origIndex'],
            variation_jer_factor = arrays.array[jets+'_jerFactorDown'],
            )
        arrays.array[jets+'.fCoordinates.fPt'] = pt
        arrays.array[jets+'.fCoordinates.fEta'] = eta
        arrays.array[jets+'.fCoordinates.fPhi'] = phi
        arrays.array[jets+'.fCoordinates.fE'] = energy

    arrays.array['MET'] = arrays.array['METDown'][:,0]
    return arrays


def apply_jec_up(arrays):
    """
    Macro to apply the JEC up variation on Arrays object.
    """
    arrays = arrays.copy()
    for jets in ['Jets', 'JetsAK8', 'JetsAK15']:
        pt, eta, phi, energy = calc_jec_variation(
            arrays.array[jets+'.fCoordinates.fPt'],
            arrays.array[jets+'.fCoordinates.fEta'],
            arrays.array[jets+'.fCoordinates.fPhi'],
            arrays.array[jets+'.fCoordinates.fE'],
            jer_factor = arrays.array[jets+'_jerFactor'],
            jec_unc = arrays.array[jets+'_jecUnc'],
            orig_idx = arrays.array[jets+'_origIndex'],
            variation_orig_idx = arrays.array[jets+'JECup_origIndex'],
            variation_jer_factor = arrays.array[jets+'JECup_jerFactor'],
            )
        arrays.array[jets+'.fCoordinates.fPt'] = pt
        arrays.array[jets+'.fCoordinates.fEta'] = eta
        arrays.array[jets+'.fCoordinates.fPhi'] = phi
        arrays.array[jets+'.fCoordinates.fE'] = energy

    arrays.array['MET'] = arrays.array['METUp'][:,1]
    return arrays


def apply_jec_down(arrays):
    """
    Macro to apply the JEC down variation on Arrays object.
    """
    arrays = arrays.copy()
    for jets in ['Jets', 'JetsAK8', 'JetsAK15']:
        pt, eta, phi, energy = calc_jec_variation(
            arrays.array[jets+'.fCoordinates.fPt'],
            arrays.array[jets+'.fCoordinates.fEta'],
            arrays.array[jets+'.fCoordinates.fPhi'],
            arrays.array[jets+'.fCoordinates.fE'],
            jer_factor = arrays.array[jets+'_jerFactor'],
            jec_unc = arrays.array[jets+'_jecUnc'],
            orig_idx = arrays.array[jets+'_origIndex'],
            variation_orig_idx = arrays.array[jets+'JECdown_origIndex'],
            variation_jer_factor = arrays.array[jets+'JECdown_jerFactor'],
            )
        arrays.array[jets+'.fCoordinates.fPt'] = pt
        arrays.array[jets+'.fCoordinates.fEta'] = eta
        arrays.array[jets+'.fCoordinates.fPhi'] = phi
        arrays.array[jets+'.fCoordinates.fE'] = energy

    arrays.array['MET'] = arrays.array['METDown'][:,1]
    return arrays
