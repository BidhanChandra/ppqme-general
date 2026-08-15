from ppqme.model import make_channel_labels
from models.singlet_fission import HAMILTONIAN_EV

def test_seven_active_channels():
    labels=make_channel_labels(HAMILTONIAN_EV)
    assert labels == [(0,0),(1,1),(2,2),(0,2),(1,2),(2,0),(2,1)]
