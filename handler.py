try:
    import unzip_requirements
except ImportError:
    pass
from src.sharpeRatioData.sharpeRatioReports import SharpeRatioReports
from src.sharpeRatioData.sharpeRecommendedHoldings import SharpeRecommendedPortfolio


def skip_execution_if_warmup_call(func):
    def wrapper(event, context):
        if event.get("source") == "serverless-plugin-warmup":
            print("WarmUp - Lambda is warm!")
            return {}
        return func(event, context)

    return wrapper


@skip_execution_if_warmup_call
def main(event, context):

    arguments = event["arguments"]
    if event["field"] == "sharpeRatioReports":
        sharpeRatioData = SharpeRatioReports(arguments["portfolio"])
        return sharpeRatioData.result

    elif event["field"] == "sharpeRecommendedPortfolio":
        sharpeRatioData = SharpeRecommendedPortfolio(arguments["portfolio"])
        return sharpeRatioData.result
    else:
        raise Exception("PYTHON MAIN HANDLER ERROR")
