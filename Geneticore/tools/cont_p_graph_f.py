import math

#Modified sigmod function, not great, but it gets the job done
def ranged_sigmoid(x, range):
  output = 0

  if x < 0:
    output = -1 * (1/(1 + math.e**x))
  else:
    output = 1/(1 + math.e**-x)

  output *= range
  return output