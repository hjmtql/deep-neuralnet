module Common (
    genWeights,
    inputWithBias,
    weightWithoutBias
  ) where

import Numeric.LinearAlgebra

genWeights :: [Int] -> IO [Matrix R]
genWeights ns = mapM genWeight ts
  where ts = zip ns (tail ns)

genWeight :: (Int, Int) -> IO (Matrix R)
genWeight (i, o) = randn o (i + 1) -- +1: bias

inputWithBias :: Matrix R -> Matrix R
inputWithBias v = v === konst 1 (1, cols v) -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias w = w ?? (All, DropLast 1)
