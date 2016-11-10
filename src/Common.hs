module Common (
    Tensor,
    Layer,
    genTensor,
    forwards,
    forward,
    sigmoid,
    dsigmoid,
    inputWithBias,
    weightWithoutBias
  ) where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

sigmoid n = 1 / (1 + exp (-n))
dsigmoid n = sigmoid n * (1 - sigmoid n)

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (uncurry randn) ts
  where ts = drop 1 . zip ns $ 0 : fmap (+1) ns -- +1: bias

forwards :: Tensor -> Matrix R -> Matrix R
forwards ws vs = foldr forward vs $ reverse ws

forward :: Matrix R -> Matrix R -> Matrix R
forward w vs = sigmoid $ w <> vbs
  where vbs = inputWithBias vs

inputWithBias :: Matrix R -> Matrix R
inputWithBias vs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias = fromLists . fmap init . toLists
