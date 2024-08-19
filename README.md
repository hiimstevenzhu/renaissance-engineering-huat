# renaissance engineering huat

A really exciting experience. 2nd in Singapore, 63rd global.

Core members:

Code writers:
Steven Zhu (hiimstevenzhu)
Asher Tam (ashertm)
Ashish Chugh (ayyshish)

Manual round solvers:
Claire Chu
Lokeshh Sampath

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

This meant that we could very simply change our strategy into exploiting this cross-exchange arbitrage, profitting off the discrepancy between the order book and the foreign exchange's price for ORCHIDS.

After simplifying our algorithm into just market making on current price, we peaked our PnLs at roughly ~92k by optimising how far in we crossed into the order book. We found out that perhaps there would have been a bit more SeaShells to be earned from market taking as well, but we could not write the algorithm for this in time.

Manual trading was a simple exercise - there were roughly at max 256 different states to find the path for the maximum arbitrage, and we almost quickly found the solution by hand, but a simple DFS into all the possible states gave us an answer that produced roughty 0.059% profit.

The final answer we had was: SHELLS -> PIZZA -> WASABI -> SHELLS -> PIZZA -> SHELLS

END RESULT: 615.7k Seashells, 150th place globally, 7th in Singapore (but we were effectively holding positions 4-6 as well)

# Round 3

Gift Baskets was essentially a simulation of ETFs. We explored several strategies, of which included comparing and trading gift baskets against its components depending on their price. We quickly realised that we were trading the components at a loss - the components acted as leading indicators for trading gift baskets. After this, we limited our trading to just gift baskets.

For manual trading, we made guesses on profitable expedition spots almost solely based off instinct. We calculated an average pay-out for each tile, and opted for sub-optimal tiles where we could expect less variance in return for a generally lower expected pay-out. We did pretty well.

# Round 4

COCONUTs and COCONUT_COUPONs were an interesting dive into options and simple greeks for us. A quick EDA indicated that the coupons' delta was near 0.5 - which meant our coupons were at the money. Since the delta was virtually constant, there was no reason for us to consider a hedging strategy - there was no gamma to scalp, and since the coupons were going to expire years from now, there was no theta to earn either. However, with a constant delta, we could model a theoretical price for the coupons depending on the price of the coconuts! We wrote up a quick Black-Scholes model, and traded our coupons around this model. Fine-tuning the parameters meant having to grid-search optimal values, but this was very possible with jmerle's visualiser. (Credits to him, without him a lot of us would not have made it this far!)

We followed our intuition for bids in the same manner as round 3, and did pretty well here as well.

# Round 5

With de-anonymised data, we started finding signals to trade every previous item. Some staring and (mostly unfruitful) EDA resulted in us trading the following commodities on certain signals:

- COCONUTs depending on Raj's position
- ROSES depending on Rihanna's position

# Finale

We ended up 63rd overall on the global boards, and a cool 2nd place within Singapore. This was a really fun experience for all of us, and we really enjoyed ourselves while learning and exploring what finance and trading had to offer.

# Credits

We'd like to thank @jmerle for his wonderful visualiser (https://github.com/jmerle/imc-prosperity-2-visualizer) and backtester (https://github.com/jmerle/imc-prosperity-2-backtester). Otherwise, we'd like to thank IMC for this amazing experience - we enjoyed our little vacation. 

  
