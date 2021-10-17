import numpy as np
import pandas as pd
from src.sharpeRatioData.main import SharpeRatioData
from libs.errorHandler import set_log

logger = set_log(__name__)


# Constant
CASH = 100000
RISK_FREE_RATE =0.02


class SharpeRatioReports(SharpeRatioData):

    def __init__(self, arguments):
        super().__init__(arguments)
        self.result = self.sharpeRatioReports()

    def sharpeRatioReports(self):
        try:
            S = self.covariance
            df = self.df
            userWeights = self.userWeights

            optimalWeights = self.optimalWeights

            userWeights = np.array(userWeights)

            optAssignedCash = optimalWeights * CASH
            optUnitHoldings = np.divide(optAssignedCash, df.iloc[0])

            userAssignedCash = userWeights * CASH
            userUnitHoldings = np.divide(userAssignedCash, df.iloc[0])

            date, userPort, optPort = [], [], []
            for i in range(len(df)):
                date.append(df.iloc[i].name)
                userPort.append(np.dot(userUnitHoldings, df.iloc[i])/CASH)
                optPort.append(np.dot(optUnitHoldings, df.iloc[i])/CASH)

            comparedResult = pd.DataFrame(
                {"Date": date, "User": userPort, "Opt": optPort})
            comparedResult = comparedResult.set_index(['Date'])
            self.comparedResult = comparedResult

            expectedReturnsMean = np.array(self.expectedReturnsMean)

            userPortfolioReturns = np.sum(expectedReturnsMean * userWeights)
            optimizedPortfolioReturns = np.sum(
                expectedReturnsMean * optimalWeights)

            userPortfolioVolatility = np.sqrt(
                np.dot(userWeights.T, np.dot(S, userWeights)))
            optimalPortfolioVolatility = np.sqrt(
                np.dot(optimalWeights.T, np.dot(S, optimalWeights)))
            userPortfolioSharpe = (userPortfolioReturns-RISK_FREE_RATE)/userPortfolioVolatility
            optimalPortfolioSharpe = (optimizedPortfolioReturns-RISK_FREE_RATE)/optimalPortfolioVolatility

            result = {"inputPortfolio": {"expectedReturnRates": [], "volatility": 0.0, "sharpeRatio": 0.0},
                      "optimalPortfolio": {"expectedReturnRates": [], "volatility": 0.0, "sharpeRatio": 0.0}}
            for i in range(len(df)):
                result["inputPortfolio"]["expectedReturnRates"].append({"date": list(comparedResult.index)[
                    i], "expectedReturnRate": list(comparedResult["User"])[i]})
                result["optimalPortfolio"]["expectedReturnRates"].append({"date": list(comparedResult.index)[
                    i], "expectedReturnRate": list(comparedResult["Opt"])[i]})

            result['inputPortfolio']['volatility'] = userPortfolioVolatility
            result['optimalPortfolio']['volatility'] = optimalPortfolioVolatility
            result['inputPortfolio']['sharpeRatio'] = round(
                userPortfolioSharpe * 10, 2)
            result['optimalPortfolio']['sharpeRatio'] = round(
                optimalPortfolioSharpe * 10, 2)
            return result
        except Exception as error:
            logger.error("PYTHON INTERNAL FUNCTION ERROR")
            raise error
