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
preTrains :: (Matrix R -> Matrix R) -> (Matrix R -> Matrix R) -> Matrix R -> [Matrix R] -> [Matrix R]
preTrains f df x ws = case ws of
  [] -> []
  m:ms -> [nm] `mappend` preTrains f df y ms
    where
      nm = preTrain f df x m
      y = forward f nm x

preTrain :: (Matrix R -> Matrix R) -> (Matrix R -> Matrix R) ->  Matrix R -> Matrix R -> Matrix R
preTrain f df x w = head . last . take 2000 $ iterate (backProp f f df x x) [w, tw]
  where
    tw = fromLists . fmap (\xs -> xs `mappend` [average xs]) $ toLists tw' -- initial bias: weight average
    tw' = tr $ weightWithoutBias w

average :: [R] -> R
average xs = sum xs / fromIntegral (length xs)
