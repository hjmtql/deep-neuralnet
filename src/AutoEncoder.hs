module AutoEncoder (
  preTrains,
  preTrain
  ) where

import Numeric.LinearAlgebra
import Common
import Forward
import BackProp
import ActivationFunction

-- TODO: refactor
preTrains :: (Matrix R -> Matrix R, Matrix R -> Matrix R) -> Matrix R -> [Matrix R] -> [Matrix R]
preTrains fAndDf x ws = fmap snd (tail (scanl (nextTrain fAndDf) (x, m) ms)) `mappend` [m]
  where (ms, m) = (init ws, last ws)

nextTrain :: (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> Matrix R -> (Matrix R, Matrix R)
nextTrain (f, df) (x, _) w = (nx, nw)
  where
    nx = forward f nw x
    [nw, _] = last . take 1000 $ iterate (preTrain (f, df) x) [w, mirrorWeight w]

preTrain :: (Matrix R -> Matrix R, Matrix R -> Matrix R) ->  Matrix R -> [Matrix R] -> [Matrix R]
preTrain (f, df) x = backProp f (f, df) (x, x)

mirrorWeight :: Matrix R -> Matrix R
mirrorWeight w = w' ||| konst 0 (rows w', 1) -- initial bias: 0
  where w' = tr $ weightWithoutBias w
