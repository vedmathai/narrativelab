
class TemporalValue:
    def __init__(self):
        self._value
        self._unit

    def value(self) -> str:
        return self._value

    def unit(self) -> str:
        return self._unit

    def set_value(self, value: str):
        self._value = value

    def set_unit(self, unit: str):
        self._unit = unit

    def to_dict(self):
        return {
            "value": self.value(),
            "unit": self.unit(),
        }

    @staticmethod
    def from_unit(val):
        temporal_value = TemporalValue()
        temporal_value.set_value(val['value'])
        temporal_value.set_unit(val['unit'])
        return temporal_value
