"""Type aliases for the powergrid case study."""

from typing import Literal


# Control mode for inverter-based sources
CtrlMode = Literal["q_set", "pf_set", "volt_var", "off"]
