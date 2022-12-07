# About

This package aims to make it easy to process data for the boosted SVJ analysis.


## Setup

```
pip install svj_ntuple_processing
```


## Example: Featurizing TreeMaker Ntuples

Open a TreeMaker Ntuple root file and get an `Arrays` object:

```python
>>> import svj_ntuple_processing as svj
>>> arrays = svj.open_root('root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.root')
```

The `Arrays` object is just a container for an `awkward.highlevel.Array` object with some other data:

```python
>>> arrays.array
<Array [{'JetsAK8.fCoordinates.fPt': [, ... ] type='60880 * {"JetsAK8.fCoordinat...'>
>>> 
>>> arrays.array['JetsAK15_axismajor']
<Array [[0.0318, 0.0528, ... 0.445, 0.709]] type='60880 * var * float32'>
>>> 
>>> arrays.metadata
{'year': 2018, 'src': 'root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.root'}
>>> 
>>> len(arrays)
60880
```

Applying the preselection and other filters is done as follows:

```python
arrays = svj.filter_preselection(arrays)
arrays = svj.filter_zprime_in_cone(arrays)
```

The resulting `Arrays` object will have much less events:

```python
>>> len(arrays)
3881
```

The `Arrays.cutflow` attribute tells you the event counts at all the intermediate stages:

```python
>>> arrays.cutflow
OrderedDict([('raw', 60880), ('ak8jet.pt>500', 12861), ('triggers', 12735), ('n_ak15jets>=2', 12735), ('subl_eta<2.4', 12669), ('subl_ecf>0', 12386), ('rtx>1.1', 7711), ('nleptons=0', 7424), ('metfilter', 7352), ('preselection', 7352), ('1zprime2darkquarks', 7263), ('zdq<1.5', 3881)])
```

At this poit the `Arrays` object still refers to literal entries of the TreeMaker Ntuple, and it is still awkwardly shaped:

```python
>>> arrays.array['JetsAK8.fCoordinates.fPt']
<Array [[1.07e+03, 742, 311], ... [705, 296]] type='3881 * var * float32'> # Sometimes 3, sometimes 2
```

Converting the `Arrays` object to a `Columns` object cuts down the TreeMaker entries to only the features we need to train a BDT:

```python
>>> columns = svj.bdt_feature_columns(arrays)
>>> columns
<svj_ntuple_processing.Columns object at 0x7fef98a14160>
```

The `Columns` object is rectangular, i.e. one number per event:

```python
>>> columns.arrays['girth']
array([0.21778949, 0.5478329 , 0.4840784 , ..., 0.35122713, 0.60851824,
       0.5864864 ], dtype=float32)
>>>
>>> columns.arrays['axismajor']
array([0.16224642, 0.33115667, 0.32601607, ..., 0.29639566, 0.50617486,
       0.6152444 ], dtype=float32)
>>>
>>> len(columns.arrays['axismajor'])
3881
>>>
>>> sorted(columns.arrays.keys())
['axismajor', 'axisminor', 'e', 'ecfc2b1', 'ecfd2b1', 'ecfm2b1', 'ecfn2b2', 'eta', 'girth', 'metdphi', 'mt', 'phi', 'pt', 'ptd', 'weight']
```

It has copies of the metadata and the cutflow:

```python
>>> columns.metadata
{'year': 2018, 'src': 'root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.root'}
>>>
>>> columns.cutflow
OrderedDict([('raw', 60880), ('ak8jet.pt>500', 12861), ('triggers', 12735), ('n_ak15jets>=2', 12735), ('subl_eta<2.4', 12669), ('subl_ecf>0', 12386), ('rtx>1.1', 7711), ('nleptons=0', 7424), ('metfilter', 7352), ('preselection', 7352), ('1zprime2darkquarks', 7263), ('zdq<1.5', 3881)])
```

It has a method to convert it to a numpy array:

```python
>>> # All features, alphabetically (first one in these case being 'axismajor')
>>> columns.to_numpy()
array([[1.62246421e-01, 7.27319717e-02, 7.94109497e+02, ...,
        2.19499037e-01, 1.26104057e+00, 1.00000000e+00],
       [3.31156671e-01, 1.11443296e-01, 4.05791321e+02, ...,
        2.52665043e-01, 1.72563887e+00, 1.00000000e+00],
       [3.26016068e-01, 6.83244318e-02, 3.56210236e+02, ...,
        2.47978821e-01, 1.56911826e+00, 1.00000000e+00],
       ...,
       [2.96395659e-01, 1.04778111e-01, 9.25817627e+02, ...,
        1.88375130e-01, 1.23072338e+00, 1.00000000e+00],
       [5.06174862e-01, 1.76295370e-01, 4.07023956e+02, ...,
        1.99949041e-01, 1.54773271e+00, 1.00000000e+00],
       [6.15244389e-01, 2.61853844e-01, 1.47918066e+03, ...,
        1.91158906e-01, 1.34534931e+00, 1.00000000e+00]], dtype=float32)
>>>
>>> # Or only use a few features, order taken from the list
>>> columns.to_numpy(['girth', 'axismajor', 'axisminor'])
array([[0.21778949, 0.16224642, 0.07273197],
       [0.5478329 , 0.33115667, 0.1114433 ],
       [0.4840784 , 0.32601607, 0.06832443],
       ...,
       [0.35122713, 0.29639566, 0.10477811],
       [0.60851824, 0.50617486, 0.17629537],
       [0.5864864 , 0.6152444 , 0.26185384]], dtype=float32)
```

And finally it can be saved and loaded to a .npz file:

```python
cols.save('cols.npz')
```


## Example: Loading featurized data

Open a file:

```python
>>> import svj_ntuple_processing as svj
>>> columns = svj.Columns.load('root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.npz')
```

Much like the the previous examples, a numpy array can be generated as follows, and the cutflow and metadata are still accessible:

```python
>>> X = columns.to_numpy(['girth', 'axismajor', 'axisminor'])
>>> X
array([[0.21778949, 0.16224642, 0.07273197],
       [0.5478329 , 0.33115667, 0.1114433 ],
       [0.4840784 , 0.32601607, 0.06832443],
       ...,
       [0.36425185, 0.1331028 , 0.10360279],
       [0.42897484, 0.33732802, 0.21320845],
       [0.17855313, 0.06017362, 0.0279889 ]], dtype=float32)
>>>
>>> X.shape
(23005, 3)
>>>
>>> columns.cutflow
OrderedDict([('raw', 362621), ('ak8jet.pt>500', 76229), ('triggers', 75436), ('n_ak15jets>=2', 75432), ('subl_eta<2.4', 75008), ('subl_ecf>0', 73337), ('rtx>1.1', 45392), ('nleptons=0', 43728), ('metfilter', 43187), ('preselection', 43187), ('1zprime2darkquarks', 42645), ('zdq<1.5', 23005)])
>>> 
>>> columns.metadata
{'year': 2018, 'madpt': 300, 'mz': 350, 'mdark': 10, 'rinv': 0.3, 'zprimecone': True, 'src': 'root://cmseos.fnal.gov//store/user/klijnsma/package_test_files/svj_ntuple_processing/madpt300_mz350_mdark10_rinv0.3.npz'}
```