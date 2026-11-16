"""Apply the chronology, serialization and live/batch contract to rich splines."""
from test_spline_ridge import SplineDemandTests
from optimise_spline_ridge import OptimisedSplineDemand


class OptimisedDemandTests(SplineDemandTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = OptimisedSplineDemand(features='rich').fit(
            cls.features.loc[cls.train], cls.truth.loc[cls.train])
