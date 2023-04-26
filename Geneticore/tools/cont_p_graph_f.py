import math

#Modified sigmod function, not great, but it gets the job done
def ranged_sigmoid(x, range):
  return (-1 * (1/(1 + math.e**x)) if x<0 else 1/(1 + math.e**-x)) * range