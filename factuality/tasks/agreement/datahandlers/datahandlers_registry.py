from factuality.tasks.agreement.datahandlers.agreement_datahandler import AgreementDatahandler


class DatahandlersRegistry:
    _dict = {
        "debagreement": AgreementDatahandler
    }

    def get_datahandler(self, name):
        handler = DatahandlersRegistry._dict[name]()
        handler.load()
        return handler
