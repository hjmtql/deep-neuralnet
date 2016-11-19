module AutoEncoder (
  preTrains,
  preTrain
  ) where

import Numeric.LinearAlgebra
import Common
import Forward
import BackProp

preTrains :: Matrix R -> [Matrix R] -> [Matrix R]
preTrains x ws = case ws of
  [] -> []
  m:ms -> [nm] `mappend` preTrains y ms
    where
      nm:_ = preTrain x m
      y = forward nm x

preTrain :: Matrix R -> Matrix R -> [Matrix R]
preTrain x w = foldr (\f x -> f x) [w, tw] (replicate 2000 $ backPropagation x x)
  where
    tw = fromLists . fmap (\xs -> xs `mappend` [average xs]) $ toLists tw' -- initial bias: weight average
    tw' = tr $ weightWithoutBias w

average :: [R] -> R
average xs = sum xs / fromIntegral (length xs)
