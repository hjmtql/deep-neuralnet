module Lib
    ( someFunc
    ) where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (\(x, y) -> uncurry randn (x, y + 1)) ts -- 1: bias
  where ts = drop 1 $ zip ns (0:ns)

forwards :: Tensor -> Matrix R -> Matrix R
forwards t vs = foldr forward vs $ reverse t

forward :: Matrix R -> Matrix R -> Matrix R
forward m vs = sigmoid $ m <> vbs
  where vbs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

sigmoid n = 1 / (1 + exp (-n))

someFunc :: IO ()
someFunc = putStrLn "someFunc"
