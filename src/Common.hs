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
inputWithBias vs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias = fromLists . fmap init . toLists
