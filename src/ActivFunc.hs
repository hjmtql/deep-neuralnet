module ActivFunc (
    sigmoidC,
    sigmoids,
    rampC,
    ramps,
    softmaxC
  ) where

import Numeric.LinearAlgebra

sigmoid x = 1 / (1 + exp (-x))
dsigmoid x = sigmoid x * (1 - sigmoid x)

sigmoidC :: Matrix R -> Matrix R
sigmoidC = cmap sigmoid

dsigmoidC :: Matrix R -> Matrix R
dsigmoidC = cmap dsigmoid

sigmoids = (sigmoidC, dsigmoidC)

ramp x = if x > 0 then x else 0
dramp x = if x > 0 then 1 else 0 -- as a matter of convenience

rampC :: Matrix R -> Matrix R
rampC = cmap ramp

drampC :: Matrix R -> Matrix R
drampC = cmap dramp

ramps = (rampC, drampC)

softmax :: Vector R -> Vector R
softmax us = cmap (/ (sum . toList) es) es
  where es = exp us

softmaxC :: Matrix R -> Matrix R
softmaxC = fromColumns . fmap softmax . toColumns
