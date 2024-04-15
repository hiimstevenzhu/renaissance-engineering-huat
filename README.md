# renaissance engineering huat

huat and prosper

# Required libraries:

1. pandas
2. numpy
3. statistics
4. math
5. typing
6. jsonpickle

# Overview:

The trading algorithm uses a standardised Trader class, which only has a single method called run. This method contains all the trading logic coded. The simulation consists of multiple iterations, and in each iteration run() is called and provided with a TradingState object. This object is the overview of all the trade that has happened in the last iteration. This TradingState is essentially a log of trades that has happened on the last tick(?).

TradingState also contains a per product overview of all the outstanding buy and sell quotes from the bots. Based on the written logic, run() can decide to send orders that fully/partially match with existing orders.

Excess orders will be left as outstanding quotes which the bots could potentially trade on. During the next iteration, TradingState reveals if any of the bots "traded" on the outstanding quotes. _Any quotes untraded on is cancelled._

(Inventory management and keeping track of orders is important)

_Trades changes the position of the Trader class in a certain asset. If the total sum of all the buy/sell orders result in a position exceeding the limit, all orders are cancelled._

# Round 1

AMETHYSTS, like pearls in the previous round, were stable at 10k, but occasionally showed some variance of +/- 1, 2, 3, 4, 5. Thus the strategy we had in mind was simple - we conducted arbitrage on these prices. Simple market making and market taking showed decent PnLs. Beyond this was just a bit of optimisation at where our threshold for favourable trades should be.

STARFRUIT was a LR problem, and we found somewhat similar PnLs from doing regression on prices and returns. Returns gave us just a little less PnL, but we figured that since it was a lot more general and showcased better stationarity doing LR on returns was a better idea.

For manual trading, the problem to solve was decently straightforward - we wanted to find bid values X and Y such that X and Y gave us the highest E(profit) where the goldfishes' reservation prices followed a linear PDF from 0 at 900 to 0.2 at 1000. Doing some simple coding (Wolfram alpha was viable as well, if you didn't want to calculate it by hand), the values we got were 952 and 978, giving us an E(profit) of ~20.4.

END RESULTS: 130.4k SeaShells, 357th place globally, 14th in Singapore.

# Round 2

This round was much less straightforward. We, like most other teams, spent a lot of time on a red herring into trying to predict ORCHIDS prices using sunlight and humidity. We didn't manage to find a suitable model for this (not yet, at least - we still think there is some sort of way to get a strong signal here we just didn't have the knoweldge to capture), and so we attempted to simplify our model into predicting prices using an LR of a modelled function for sunlight and humdity somewhat as follows:

    f(sunlight/humidity, within_range) = 1 if sunlight/humidity is within_range else deviation ** 2

We thought we performed well by shorting ORCHIDS based on our prediction, but we soon found out that our LR model effectly turned into an AR(1) model on previous prices from our coefficients. Our algo worked well at first, generating ~67k profits on maximum, but we eventually found out after some tracing that what we were essentially doing is doing maxed position shorts on ORCHIDS, earning from the arbitrage.

After simplifying our algorithm into just market making on current price, we peaked our PnLs at roughly ~92k by optimising how far in we crossed into the order book. We found out that perhaps there would have been a bit more SeaShells to be earned from market taking as well, but we could not write the algorithm for this in time.

Manual trading was a simple exercise - there were roughly at max 256 different states to find the path for the maximum arbitrage, and we almost quickly found the solution by hand, but a simple DFS into all the possible states gave us an answer that produced roughty 0.059% profit.

The final answer we had was: SHELLS -> PIZZA -> WASABI -> SHELLS -> PIZZA -> SHELLS

END RESULT: 615.7k Seashells, 150th place globally, 7th in Singapore (but we were effectively holding positions 4-6 as well)

# Round 3
