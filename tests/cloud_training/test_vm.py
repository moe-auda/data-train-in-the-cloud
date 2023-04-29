import platform
from taxifare.params import INSTANCE

def test_i_am_a_vm():
    """
    Test that this code is being run from a Google VM named as per env variable 'INSTANCE'
    """

    assert platform.node() == INSTANCE, f"You should be running from your instance named '{INSTANCE}'."
