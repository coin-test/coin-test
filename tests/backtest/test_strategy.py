"""Test the Strategy class."""

# from coin_test.backtest import Strategy


def test_strategy_valid() -> None:
    """Implement correctly."""
    pass


def test_strategy_invalid() -> None:
    """Error on missing methods."""
    pass


# class TestStrategy(Strategy):
#     """Test Strategy abstract implementation."""

#     def __init__(self) -> None:
#         """Initialize a TestStrategy object."""
#         super().__init__(
#             name="Pro Strat",
#             asset_pairs=[AssetPair(Ticker("BTC"), Ticker("USDT"))],
#             schedule="* * * * *",
#             lookback=dt.timedelta(days=5),
#         )

#     def __call__(
#         self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
#     ) -> list[TradeRequest]:
#         """Execute test strategy."""
#         asset_pair = self.asset_pairs[0]

#         if portfolio.available_assets(Ticker("BTC")) == Money(Ticker("BTC"), 0):
#             # if no holdings in bitcoin, go all in
#             x = MarketTradeRequest(
#                 asset_pair,
#                 Side.BUY,
#                 notional=portfolio.available_assets(Ticker("USDT")).qty,
#             )
#         else:
#             # otherwise sell all bitcoin holdings
#             x = MarketTradeRequest(
#      asset_pair, Side.SELL, qty=portfolio.available_assets(Ticker("BTC")).qty
#             )

#         return [x]
