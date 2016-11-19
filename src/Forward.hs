module Forward (
    forwardClassification,
    forwardRegressions,
    forward
  ) where

import Numeric.LinearAlgebra
import Common
import ActivationFunction

forwardClassification :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> Matrix R
forwardClassification f ws v = softmaxC $ forwardRegressions f ws v

forwardRegressions :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> Matrix R
forwardRegressions f ws v = m <> inputWithBias nv
  where
    nv = foldl (flip (forward f)) v ms
    (ms, m) = (init ws, last ws)

forward :: (Matrix R -> Matrix R) -> Matrix R -> Matrix R -> Matrix R
forward f w v = f $ w <> bv
  where bv = inputWithBias v
