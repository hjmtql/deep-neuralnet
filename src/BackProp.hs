module BackProp (
    backPropRegression,
    backPropClassification,
    backProp,
    sgdMethod
  ) where

import Numeric.LinearAlgebra
import Common
import ActivationFunction

sgdMethod :: Int -> (Matrix R, Matrix R) -> ((Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]) -> [Matrix R] -> IO [Matrix R]
sgdMethod n xy f ws = do
  nxy <- pickupSets n xy
  return $ f nxy ws

backPropRegression :: (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backPropRegression = backProp id

backPropClassification :: (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backPropClassification = backProp softmaxC

backProp :: (Matrix R -> Matrix R) -> (Matrix R -> Matrix R, Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> [Matrix R] -> [Matrix R]
backProp af (f, df) (x, y) ws@(m:ms) = zipWith (-) ws (fmap (0.1 *) dws)
  where
    dws = fmap (/ len) . zipWith (<>) ds $ fmap (tr . inputWithBias) vs
    len = fromIntegral $ cols x
    vs = x : fmap f rs -- length: L - 1
    ds = calcDeltas df (zip rs ms) dInit -- length: L - 1
    dInit = af r - y
    (rs, r) = (init us, last us)
    us = forwardUs f ms uInit -- length: L - 1
    uInit = m <> inputWithBias x

forwardUs :: (Matrix R -> Matrix R) -> [Matrix R] -> Matrix R -> [Matrix R]
forwardUs f ws u = scanl (flip (forwardU f)) u ws

forwardU :: (Matrix R -> Matrix R) -> Matrix R -> Matrix R -> Matrix R
forwardU f w u = w <> inputWithBias (f u)

calcDeltas :: (Matrix R -> Matrix R) -> [(Matrix R, Matrix R)] -> Matrix R -> [Matrix R]
calcDeltas df uws d = scanr (calcDelta df) d uws

calcDelta :: (Matrix R -> Matrix R) -> (Matrix R, Matrix R) -> Matrix R -> Matrix R
calcDelta df (u, w) d = df u * tr (weightWithoutBias w) <> d
