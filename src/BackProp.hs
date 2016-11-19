module BackProp (
    backPropagation
  ) where

import Numeric.LinearAlgebra
import Common
import ActivationFunction

backPropagation :: Matrix R -> Matrix R -> [Matrix R] -> [Matrix R]
backPropagation xs ys ws = zipWith (-) ws (fmap (*0.1) dws)
  where
    dws = (/ len) <$> zipWith (<>) ds (fmap (tr . inputWithBias) (init vs))
    ds = reverse $ calcDelta (reverse (zip (init us) (tail ws))) dInit -- length: L - 1
    dInit = last vs - ys
    us = forwardUs (tail ws) uInit -- length: L - 1
    uInit = head ws <> inputWithBias xs
    vs = xs : fmap sigmoidC us -- length: L
    len = fromIntegral $ cols xs

forwardUs :: [Matrix R] -> Matrix R -> [Matrix R]
forwardUs ws u = [u] `mappend` case ws of
  [] -> []
  m:ms -> forwardUs ms nu
    where nu = forwardU m u

forwardU :: Matrix R -> Matrix R -> Matrix R
forwardU w u = w <> inputWithBias v
  where v = sigmoidC u

calcDelta :: [(Matrix R, Matrix R)] -> Matrix R -> [Matrix R]
calcDelta uws d = [d] `mappend` case uws of
  [] -> []
  (u, w):ts -> calcDelta ts nd
    where nd = dsigmoidC u * tr (weightWithoutBias w) <> d
