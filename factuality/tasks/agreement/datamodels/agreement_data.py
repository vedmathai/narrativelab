from factuality.tasks.agreement.datamodels.agreement_datum import AgreementDatum


class AgreementData:
    def __init__(self):
        self._data = []

    def add_datum(self, datum):
        return self._data.append(datum)
    
    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data

    @staticmethod
    def from_dict(val):
        agreement_data = AgreementData()
        agreement_data.set_data(AgreementDatum.from_dict(i) for i in val['agreement_data'])
        return agreement_data
    
    def to_dict(self):
        return {
            'agreement_data': [i.to_dict() for i in self.data()]
        }
