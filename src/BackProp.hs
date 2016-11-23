module BackProp (
    backPropRegression,
    backPropClassification,
    backProp,
    sgdMethod
  ) where

import Numeric.LinearAlgebra
import Common
import ActivFunc

sgdMethod :: Int -> (Matrix R, Matrix R) -> ((Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]) -> [Matrix R] -> IO [Matrix R]
sgdMethod n xy f ws = do
  nxy <- pickupSets n xy
  return $ f nxy ws

backPropRegression :: R -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backPropRegression = backProp id

backPropClassification :: R -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backPropClassification = backProp softmaxC

backProp :: (Matrix R -> Matrix R) -> R -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backProp lf rate (f, df) (x, y) ws@(m:ms) = zipWith (-) ws (fmap (scalar rate *) dws)
  where
    dws = fmap (/ len) . zipWith (<>) ds $ fmap (tr . inputWithBias) vs
    len = fromIntegral $ cols x
    vs = x : fmap f rs -- length: L - 1
    ds = calcDeltas df (zip rs ms) dInit -- length: L - 1
    dInit = lf r - y
    (rs, r) = (init us, last us)
    us = forwards' f ms uInit -- length: L - 1
    uInit = m <> inputWithBias x

forwards' :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> [Matrix R]
forwards' f ws u = scanl (flip $ forward' f) u ws

forward' :: (Matrix R -> Matrix R) -> Matrix R -> Matrix R -> Matrix R
forward' f w u = w <> inputWithBias (f u)

calcDeltas :: (Matrix R -> Matrix R) -> [(Matrix R, Matrix R)] -> Matrix R -> [Matrix R]
calcDeltas df uws d = scanr (calcDelta df) d uws

calcDelta :: (Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> Matrix R -> Matrix R
calcDelta df (u, w) d = df u * tr (weightWithoutBias w) <> d
