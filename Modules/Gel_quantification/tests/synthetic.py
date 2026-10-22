"""The synthetic gels live in the package now (gelanalyzer.synthetic), so the
run script's mock example can use them too. Tests keep importing from here."""

from gelanalyzer.synthetic import *  # noqa: F401,F403
from gelanalyzer.synthetic import LADDER_FRACTIONS, LADDER_THICK, Band, SyntheticGel, make_gel  # noqa: F401
