module Forward (
    forwards,
    forward
  ) where

import Numeric.LinearAlgebra
import Common
import ActivationFunction

forwards :: [Matrix R] -> Matrix R -> Matrix R
forwards ws vs = foldr forward vs $ reverse ws

forward :: Matrix R -> Matrix R -> Matrix R
forward w vs = sigmoidC $ w <> vbs
  where vbs = inputWithBias vs
