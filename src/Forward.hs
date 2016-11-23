module Forward (
    forwardClassification,
    forwardRegression,
    forward
  ) where

import Numeric.LinearAlgebra
import Common
import ActivFunc

forwardClassification :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> Matrix R
forwardClassification f ws v = softmaxC $ forwardRegression f ws v

forwardRegression :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> Matrix R
forwardRegression f ws v = forward id (last ws) nv
  where nv = forwards f (init ws) v

forwards :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> Matrix R
forwards f ws v = foldl (flip $ forward f) v ws

forward :: (Matrix R -> Matrix R) -> Matrix R -> Matrix R -> Matrix R
forward f w v = f $ w <> inputWithBias v
