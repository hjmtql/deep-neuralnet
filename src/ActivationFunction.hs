module ActivationFunction (
    sigmoid,
    dsigmoid,
    rampC,
    drampC,
    softmaxC
  ) where

import Numeric.LinearAlgebra

sigmoid x = 1 / (1 + exp (-x))
dsigmoid x = sigmoid x * (1 - sigmoid x)

ramp :: R -> R
ramp = max 0

dramp :: R -> R
dramp x = if x > 0 then 1 else 0 -- as a matter of convenience

rampC :: Matrix R -> Matrix R
rampC = cmap ramp

drampC :: Matrix R -> Matrix R
drampC = cmap dramp

softmaxC :: Matrix R -> Matrix R
softmaxC = fromColumns . fmap softmax . toColumns

softmax :: Vector R -> Vector R
softmax us = cmap (/ (sum . toList) es) es
  where es = exp us
