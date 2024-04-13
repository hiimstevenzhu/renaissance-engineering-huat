storex = storey = maxprofit = 0
for x in range(0,101):
    for y in range(x,101):
        currentprofit = (x/100)**2 * (100-x) + ((y/100)**2 -(x/100)**2)* (100-y)
        if currentprofit > maxprofit:
            storex = x
            storey = y
            maxprofit = currentprofit

print("this is x", storex)
print("this is y", storey)
print("this is maxprofit", maxprofit)
