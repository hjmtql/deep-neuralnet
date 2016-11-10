module BackProp (
    backPropagation
  ) where

import Common
import Numeric.LinearAlgebra

backPropagation :: Matrix R -> Matrix R -> Tensor -> Tensor
backPropagation xs ys ws = zipWith (-) ws (fmap (*0.1) dws)
  where
    dws = (/ len) <$> zipWith (<>) ds (fmap (tr . inputWithBias) (init vs))
    ds = reverse $ calcDelta (reverse (zip (init us) (tail ws))) dInit -- length: L - 1
    dInit = last vs - ys
    us = forwardUs (tail ws) uInit -- length: L - 1
    uInit = head ws <> inputWithBias xs
    vs = xs : fmap sigmoid us -- length: L
    len = fromIntegral $ cols xs

forwardUs :: Tensor -> Matrix R -> Tensor
forwardUs ws u = [u] `mappend` case ws of
  [] -> []
  m:ms -> forwardUs ms nu
    where nu = forwardU m u

forwardU :: Matrix R -> Matrix R -> Matrix R
forwardU w u = w <> inputWithBias v
  where v = sigmoid u

calcDelta :: [(Matrix R, Matrix R)] -> Matrix R -> Tensor
calcDelta uws d = [d] `mappend` case uws of
  [] -> []
  (u, w):ts -> calcDelta ts nd
    where nd = dsigmoid u * tr (weightWithoutBias w) <> d
