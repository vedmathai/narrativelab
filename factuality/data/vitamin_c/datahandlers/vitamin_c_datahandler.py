from eventvec.server.data.torque.datahandlers.torque_converter import VitaminCConverter
from eventvec.server.data.torque.readers.torque_datareader import VitaminCDataReader


class VitaminCDatahandler:
    def __init__(self):
        self._vitamin_c_converter = VitaminCConverter()
        self._vitamin_c_datareader = VitaminCDataReader()

    def claims_data(self):
        vitamin_c_documents = self._vitamin_c_datareader.vitamin_c_train_dataset()
        qa_data = self._vitamin_c_converter.convert(vitamin_c_documents)
        return qa_data

    def claims_eval_data(self):
        vitamin_c_documents = self._vitamin_c_datareader.vitamin_c_eval_dataset()
        qa_data = self._vitamin_c_converter.convert(vitamin_c_documents)
        return qa_data
