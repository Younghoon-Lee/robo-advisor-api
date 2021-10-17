from src.sharpeRatioData.main import SharpeRatioData
from libs.errorHandler import set_log

logger = set_log(__name__)


class SharpeRecommendedPortfolio(SharpeRatioData):

    def __init__(self, arguments):
        super().__init__(arguments)
        self.result = self.sharpeRecommendedHoldings()

    def sharpeRecommendedHoldings(self):
        try:
            cash = self.totalCashAmount
            optimalWeights = self.optimalWeights

            allocatedCash = [cash*x for x in optimalWeights]

            holdings, recommendedCash = [], []
            for i in range(len(self.unitAmount)):
                holdings.append(allocatedCash[i]//self.unitAmount[i])
                calculated_amount = holdings[i]*self.unitAmount[i]
                if calculated_amount < self.defaultAmount[i]:
                    print("stock {} minimum bound Warning: {} is lower than {} ".format(
                        self.assets_[i], calculated_amount, self.defaultAmount[i]))
                    calculated_amount = 0
                recommendedCash.append(calculated_amount)
            print("holdings(0.01): ", holdings)
            recommendedCash.append(cash - sum(recommendedCash))
            print("User input Cash Amount: {}".format(recommendedCash))

            sharpePortfolio = {"items": []}
            for i in range(len(self.assets_)):
                sharpePortfolio["items"].append(
                    {"amount": recommendedCash[i], "stockId": self.assets_[i]})

            sharpePortfolio["cashAmount"] = recommendedCash[-1]

            return sharpePortfolio

        except Exception as error:
            logger.error("PYTHON INTERNAL FUNCTION ERROR")
            raise error
