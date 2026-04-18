import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ):
        """
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiration (in years)
        risk_free_rate: Annual risk-free interest rate (decimal, e.g. 0.05)
        volatility: Annual volatility (decimal, e.g. 0.2)
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def _d1(self):
        return (
            np.log(self.spot_price / self.strike_price)
            + (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_expiry
        ) / (self.volatility * np.sqrt(self.time_to_expiry))

    def _d2(self):
        return self._d1() - self.volatility * np.sqrt(self.time_to_expiry)

    def call_price(self):
        d1 = self._d1()
        d2 = self._d2()
        return (
            self.spot_price * norm.cdf(d1)
            - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2)
        )

    def put_price(self):
        d1 = self._d1()
        d2 = self._d2()
        return (
            self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(-d2)
            - self.spot_price * norm.cdf(-d1)
        )

    def vega(self):
        """
        Sensitivity of option price to volatility
        """
        d1 = self._d1()
        return self.spot_price * norm.pdf(d1) * np.sqrt(self.time_to_expiry)


def implied_volatility(
    market_option_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    initial_volatility_guess: float = 0.2,
    tolerance: float = 1e-6,
    max_iterations: int = 100
):
    """
    Calculate implied volatility from market price

    option_type: "call" or "put"
    """

    volatility = initial_volatility_guess

    # 🔹 Newton-Raphson Method
    for _ in range(max_iterations):
        model = BlackScholes(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility
        )

        if option_type == "call":
            model_price = model.call_price()
        else:
            model_price = model.put_price()

        vega = model.vega()
        price_error = model_price - market_option_price

        if abs(price_error) < tolerance:
            return volatility

        if vega < 1e-8:
            break  # Avoid instability

        volatility -= price_error / vega

        if volatility <= 0:
            volatility = 1e-4

    # 🔹 Fallback: Bisection Method (more stable, slower)
    lower_bound = 1e-6
    upper_bound = 5.0

    for _ in range(max_iterations):
        volatility = (lower_bound + upper_bound) / 2

        model = BlackScholes(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility
        )

        if option_type == "call":
            model_price = model.call_price()
        else:
            model_price = model.put_price()

        if abs(model_price - market_option_price) < tolerance:
            return volatility

        if model_price > market_option_price:
            upper_bound = volatility
        else:
            lower_bound = volatility

    return volatility


# 🔹 Example Usage
if __name__ == "__main__":
    spot_price = 100
    strike_price = 100
    time_to_expiry = 1.0  # 1 year
    risk_free_rate = 0.05
    volatility = 0.2

    model = BlackScholes(
        spot_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility
    )

    call_price = model.call_price()
    put_price = model.put_price()

    print("Call Price:", call_price)
    print("Put Price:", put_price)

    # Assume we observe this price in the market
    observed_market_price = call_price

    implied_vol = implied_volatility(
        observed_market_price,
        spot_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        option_type="call"
    )

    print("Implied Volatility:", implied_vol)