from datamodel import UserId, Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import jsonpickle
import numpy as np
import math
import json
from numpy.polynomial.polynomial import Polynomial

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out



class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.KELP: {
        "take_width": 1.5,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "reversion_beta": -0.1,
        "disregard_edge": 1.5,
        "join_edge": 0.5,
        "default_edge": 1.5,
    },
    Product.PICNIC_BASKET1: {
        "long_threshold": -100,
        "short_threshold": 150,
    },
    Product.PICNIC_BASKET2: {
    },
    Product.CROISSANTS: {
        "long_threshold": -100,
        "short_threshold": 150,
    },
    Product.JAMS: {
        "long_threshold": -100,
        "short_threshold": 150,
    },
    Product.DJEMBES: {
        "long_threshold": -100,
        "short_threshold": 150,
    },
    Product.VOLCANIC_ROCK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 30,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "fair_value": 500,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "fair_value": 250,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "fair_value": 250,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "fair_value": 50,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "fair_value": 250,
        "take_width": 0.8,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1.5,  # joins orders within this edge
        "default_edge": 3,
        "soft_position_limit": 10,
    },
}

def bisection(f, a, b, tol=1e-5, max_iter=1000):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for _ in range(max_iter):
        c = (a + b) / 2
        if np.abs(f(c)) < tol or (b - a) / 2 < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    raise RuntimeError("Maximum number of iterations reached")

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def implied_volatility(V, S, K, T, r=0.0, option_type="call"):
    try:
        return bisection(lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - V, 1e-5, 3.0)
    except ValueError:
        return None


logger = Logger()



class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50, Product.KELP: 50, Product.CROISSANTS: 250, Product.JAMS: 350,
        Product.DJEMBES: 60, Product.PICNIC_BASKET1: 60, Product.PICNIC_BASKET2: 100 , Product.VOLCANIC_ROCK: 400, Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
        Product.VOLCANIC_ROCK_VOUCHER_9750: 200, Product.VOLCANIC_ROCK_VOUCHER_10000: 200, Product.VOLCANIC_ROCK_VOUCHER_10250: 200, Product.VOLCANIC_ROCK_VOUCHER_10500: 200}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    def VOLCANIC_ROCK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("VOLCANIC_ROCK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["VOLCANIC_ROCK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("VOLCANIC_ROCK_last_price", None) != None:
                last_price = traderObject["VOLCANIC_ROCK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.VOLCANIC_ROCK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def VOLCANIC_ROCK_VOUCHER_10500_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("VOLCANIC_ROCK_VOUCHER_10500_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["VOLCANIC_ROCK_VOUCHER_10500_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("VOLCANIC_ROCK_VOUCHER_10500_last_price", None) != None:
                last_price = traderObject["VOLCANIC_ROCK_VOUCHER_10500_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["VOLCANIC_ROCK_VOUCHER_10500_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2


    def compute_vol_surface_parabola(self, state, S_t, TTE):
        voucher_strikes = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500,
        }

        points = []
        for product, K in voucher_strikes.items():
            od = state.order_depths.get(product, None)
            if not od or not od.buy_orders or not od.sell_orders:
                continue
            V_t = (max(od.buy_orders) + min(od.sell_orders)) / 2
            m_t = np.log(K / S_t) / np.sqrt(TTE)
            v_t = implied_volatility(V_t, S_t, K, TTE)
            if v_t is not None:
                points.append((m_t, v_t))

        if len(points) < 3:
            return None, None  # Not enough data for fit

        m_vals, v_vals = zip(*points)
        coefs = np.polyfit(m_vals, v_vals, 2)
        poly = np.poly1d(coefs)
        base_iv = poly(0)

        logger.print(f"Fitted parabola: v(m) = {coefs[0]:.4f} m² + {coefs[1]:.4f} m + {coefs[2]:.4f}")
        logger.print(f"Base IV: {base_iv:.4f}")

        return poly, points

    def sell_vol_if_overpriced(self, product, K, S_t, TTE, poly, state, result, threshold=0.03):
        order_depth = state.order_depths.get(product)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return

        if product in state.position:
            position = state.position[product]
        else:
            position = 0

        quantity = self.params[product]["soft_position_limit"] - position
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        V_t = (best_bid + best_ask) / 2

        m_t = np.log(K / S_t) / np.sqrt(TTE)
        v_market = implied_volatility(V_t, S_t, K, TTE)
        v_model = poly(m_t) if poly else None

        if v_market and v_model and (v_market - v_model > threshold):
            if best_ask - best_bid <= 2:
                price_to_sell = best_bid
            else:
                price_to_sell = best_ask - 1

            result.setdefault(product, []).append(Order(product, price_to_sell, -quantity))

        if v_market and v_model and (v_model - v_market > threshold):
                # Buy decision
                if best_ask - best_bid <= 2:
                    price_to_buy = best_ask  # lift the ask
                else:
                    price_to_buy = best_bid + 1  # penny up

                result.setdefault(product, []).append(Order(product, price_to_buy, quantity))


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            RAINFOREST_RESIN_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            RAINFOREST_RESIN_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    RAINFOREST_RESIN_position,
                )
            )
            RAINFOREST_RESIN_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    RAINFOREST_RESIN_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            RAINFOREST_RESIN_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                RAINFOREST_RESIN_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                RAINFOREST_RESIN_take_orders + RAINFOREST_RESIN_clear_orders + RAINFOREST_RESIN_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (
                SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.PICNIC_BASKET1 in self.params and Product.PICNIC_BASKET1 in state.order_depths and \
            Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths and \
            Product.JAMS in self.params and Product.JAMS in state.order_depths and \
            Product.DJEMBES in self.params and Product.DJEMBES in state.order_depths:

            croissants = self.get_mid_price(state, "CROISSANTS")
            jams = self.get_mid_price(state, "JAMS")
            djembes = self.get_mid_price(state, "DJEMBES")
            picnic_basket1 = self.get_mid_price(state, "PICNIC_BASKET1")

            diff1 = picnic_basket1 - 6 * croissants - 3 * jams - 1 * djembes

            # if diff1 < 0: buy other products, sell picnic basket


            for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1]:
                long_threshold = self.params[product]["long_threshold"]
                short_threshold = self.params[product]["short_threshold"]
                # diff is smaller than long_threshold the
                # we want to buy picnic basket, sell others
                if diff1 < long_threshold:
                    if product != Product.PICNIC_BASKET1:
                        order_depth = state.order_depths[product]
                        price = min(order_depth.buy_orders.keys())

                        position = state.position.get(product, 0)
                        to_sell = self.LIMIT[product] + position

                        result[product] = [Order(product, price, -to_sell)]
                    else:
                        order_depth = state.order_depths[product]
                        price = max(order_depth.sell_orders.keys())

                        position = state.position.get(product, 0)
                        to_buy = self.LIMIT[product] - position

                        result[product] = [Order(product, price, to_buy)]
                # diff is greater than short_threshold, we want to sell picnic basket, buy others
                elif diff1 > short_threshold:
                    if product != Product.PICNIC_BASKET1:
                        order_depth = state.order_depths[product]
                        price = max(order_depth.sell_orders.keys())

                        position = state.position.get(product, 0)
                        to_buy = self.LIMIT[product] - position

                        result[product] = [Order(product, price, to_buy)]
                    else:
                        order_depth = state.order_depths[product]
                        price = min(order_depth.buy_orders.keys())

                        position = state.position.get(product, 0)
                        to_sell = self.LIMIT[product] + position

                        result[product] = [Order(product, price, -to_sell)]

        if Product.VOLCANIC_ROCK in self.params and Product.VOLCANIC_ROCK in state.order_depths:
            VOLCANIC_ROCK_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            VOLCANIC_ROCK_fair_value = self.VOLCANIC_ROCK_fair_value(
                state.order_depths[Product.VOLCANIC_ROCK], traderObject
            )
            VOLCANIC_ROCK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    VOLCANIC_ROCK_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["take_width"],
                    VOLCANIC_ROCK_position,
                    self.params[Product.VOLCANIC_ROCK]["prevent_adverse"],
                    self.params[Product.VOLCANIC_ROCK]["adverse_volume"],
                )
            )
            VOLCANIC_ROCK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    VOLCANIC_ROCK_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["clear_width"],
                    VOLCANIC_ROCK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            VOLCANIC_ROCK_make_orders, _, _ = self.make_orders(
                Product.VOLCANIC_ROCK,
                state.order_depths[Product.VOLCANIC_ROCK],
                VOLCANIC_ROCK_fair_value,
                VOLCANIC_ROCK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.VOLCANIC_ROCK]["disregard_edge"],
                self.params[Product.VOLCANIC_ROCK]["join_edge"],
                self.params[Product.VOLCANIC_ROCK]["default_edge"],
            )
            result[Product.VOLCANIC_ROCK] = (
                VOLCANIC_ROCK_take_orders + VOLCANIC_ROCK_clear_orders + VOLCANIC_ROCK_make_orders
            )

        if Product.VOLCANIC_ROCK_VOUCHER_9500 in self.params and Product.VOLCANIC_ROCK_VOUCHER_9500 in state.order_depths:
            VOLCANIC_ROCK_VOUCHER_9500_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if Product.VOLCANIC_ROCK_VOUCHER_9500 in state.position
                else 0
            )
            VOLCANIC_ROCK_VOUCHER_9500_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.VOLCANIC_ROCK_VOUCHER_9500,
                    state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                    self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["fair_value"],
                    self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["take_width"],
                    VOLCANIC_ROCK_VOUCHER_9500_position,
                )
            )
            VOLCANIC_ROCK_VOUCHER_9500_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.VOLCANIC_ROCK_VOUCHER_9500,
                    state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                    self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["fair_value"],
                    self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["clear_width"],
                    VOLCANIC_ROCK_VOUCHER_9500_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            VOLCANIC_ROCK_VOUCHER_9500_make_orders, _, _ = self.make_orders(
                Product.VOLCANIC_ROCK_VOUCHER_9500,
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["fair_value"],
                VOLCANIC_ROCK_VOUCHER_9500_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["disregard_edge"],
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["join_edge"],
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["default_edge"],
                True,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["soft_position_limit"],
            )
            result[Product.VOLCANIC_ROCK_VOUCHER_9500] = (
                VOLCANIC_ROCK_VOUCHER_9500_take_orders + VOLCANIC_ROCK_VOUCHER_9500_clear_orders + VOLCANIC_ROCK_VOUCHER_9500_make_orders
            )

        # if Product.VOLCANIC_ROCK_VOUCHER_10500 in self.params and Product.VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths:
        #     VOLCANIC_ROCK_VOUCHER_10500_position = (
        #         state.position[Product.VOLCANIC_ROCK_VOUCHER_10500]
        #         if Product.VOLCANIC_ROCK_VOUCHER_10500 in state.position
        #         else 0
        #     )
        #     VOLCANIC_ROCK_VOUCHER_10500_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.VOLCANIC_ROCK_VOUCHER_10500,
        #             state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
        #             self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["fair_value"],
        #             self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["take_width"],
        #             VOLCANIC_ROCK_VOUCHER_10500_position,
        #         )
        #     )
        #     VOLCANIC_ROCK_VOUCHER_10500_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.VOLCANIC_ROCK_VOUCHER_10500,
        #             state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
        #             self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["fair_value"],
        #             self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["clear_width"],
        #             VOLCANIC_ROCK_VOUCHER_10500_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     VOLCANIC_ROCK_VOUCHER_10500_make_orders, _, _ = self.make_orders(
        #         Product.VOLCANIC_ROCK_VOUCHER_10500,
        #         state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
        #         self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["fair_value"],
        #         VOLCANIC_ROCK_VOUCHER_10500_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["disregard_edge"],
        #         self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["join_edge"],
        #         self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["default_edge"],
        #         True,
        #         self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["soft_position_limit"],
        #     )
        #     result[Product.VOLCANIC_ROCK_VOUCHER_10500] = (
        #         VOLCANIC_ROCK_VOUCHER_10500_take_orders + VOLCANIC_ROCK_VOUCHER_10500_clear_orders + VOLCANIC_ROCK_VOUCHER_10500_make_orders
        #     )

        # # === Bear Call Credit Spread: Sell 9750, Buy 10000 ===
        sell_sym = Product.VOLCANIC_ROCK_VOUCHER_9750
        buy_sym = Product.VOLCANIC_ROCK_VOUCHER_10000
        sell_strike = 9750
        buy_strike = 10000

        if sell_sym in state.order_depths and buy_sym in state.order_depths:
            sell_od = state.order_depths[sell_sym]
            buy_od = state.order_depths[buy_sym]

            if sell_od.buy_orders and buy_od.sell_orders:
                sell_bid = max(sell_od.buy_orders.keys())  # We sell to best bidder
                buy_ask = min(buy_od.sell_orders.keys())   # We buy from best seller
                net_credit = sell_bid - buy_ask
                max_loss = buy_strike - sell_strike - net_credit

                if net_credit > 0 and max_loss > 0:
                    qty = 20
                    result.setdefault(sell_sym, []).append(Order(sell_sym, sell_bid, -qty))  # SELL low strike
                    result.setdefault(buy_sym, []).append(Order(buy_sym, buy_ask, qty))      # BUY high strike
                    logger.print(f"Opening Bear Call Credit Spread: SELL {sell_sym}@{sell_bid}, BUY {buy_sym}@{buy_ask}, credit: {net_credit}")


        # # === Bull Put Credit Spread: Sell 10250, Buy 10000 ===
        # sell_sym = Product.VOLCANIC_ROCK_VOUCHER_10250
        # buy_sym = Product.VOLCANIC_ROCK_VOUCHER_10000
        # sell_strike = 10250
        # buy_strike = 10000

        # if sell_sym in state.order_depths and buy_sym in state.order_depths:
        #     sell_od = state.order_depths[sell_sym]
        #     buy_od = state.order_depths[buy_sym]

        #     if sell_od.buy_orders and buy_od.sell_orders:
        #         sell_bid = max(sell_od.buy_orders.keys())  # Sell high strike put
        #         buy_ask = min(buy_od.sell_orders.keys())   # Buy low strike put
        #         net_credit = sell_bid - buy_ask
        #         max_loss = sell_strike - buy_strike - net_credit

        #         if net_credit > 0 and max_loss > 0:
        #             qty = 5
        #             result.setdefault(sell_sym, []).append(Order(sell_sym, sell_bid, -qty))
        #             result.setdefault(buy_sym, []).append(Order(buy_sym, buy_ask, qty))
        #             logger.print(f"Opening Bull Put Credit Spread: SELL {sell_sym}@{sell_bid}, BUY {buy_sym}@{buy_ask}, credit: {net_credit}")

        # === Volatility Surface Analysis ===
        S_t = 10000
        if Product.VOLCANIC_ROCK in self.params and Product.VOLCANIC_ROCK in state.order_depths:
            S_t = VOLCANIC_ROCK_fair_value
        TTE = 4 / 365  # Assuming options expire in 7 days; adjust accordingly
        poly, iv_points = self.compute_vol_surface_parabola(state, S_t, TTE)

        voucher_strikes = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500,
        }

        for product, K in voucher_strikes.items():
            self.sell_vol_if_overpriced(product, K, S_t, TTE, poly, state, result)


        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
