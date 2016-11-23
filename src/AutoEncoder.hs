module AutoEncoder (
  preTrains,
  preTrain
  ) where

import Numeric.LinearAlgebra
import Data.List
import Common
import Forward
import BackProp
import ActivFunc
import Other

-- TODO: refactor
preTrains :: R -> Int -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> Matrix R -> [Matrix R] -> [Matrix R]
preTrains rate iter fDf x ws = snd (mapAccumL (preTrain rate iter fDf) x ms) `mappend` [m]
  where (ms, m) = (init ws, last ws)

preTrain :: R -> Int -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> Matrix R -> Matrix R -> (Matrix R, Matrix R)
preTrain rate iter fDf@(f, df) x w = (nx, nw)
  where
    nx = forward f nw x
    [nw, _] = last . take iter $ iterate (backProp f rate fDf (x, x)) [w, mirrorWeight w]

mirrorWeight :: Matrix R -> Matrix R
mirrorWeight w = w' ||| konst 0 (rows w', 1) -- initial bias: 0
  where w' = tr $ weightWithoutBias w
