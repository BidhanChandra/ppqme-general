import numpy as np
from ppqme.transform import PartialPolaronTransform, weighting_function

def test_redfield_limit():
    w=np.linspace(.001,.4,10)
    assert np.allclose(weighting_function(w,PartialPolaronTransform(kind='none')),0)

def test_partial_limits():
    tr=PartialPolaronTransform(omega_h=.31,alpha=2)
    assert weighting_function(np.array([1e-8]),tr)[0] < 1e-10
    assert weighting_function(np.array([10.0]),tr)[0] > .999999
