module Common (
    Tensor,
    Layer,
    genTensor,
    inputWithBias,
    weightWithoutBias
  ) where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (uncurry randn) ts
  where ts = drop 1 . zip ns $ 0 : fmap (+1) ns -- +1: bias

inputWithBias :: Matrix R -> Matrix R
inputWithBias vs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias = fromLists . fmap init . toLists
