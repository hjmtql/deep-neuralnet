module Lib
    ( someFunc
    ) where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (\(x, y) -> uncurry randn (x, y)) ts
  where ts = drop 1 $ zip ns (0:ns)

forwards :: Tensor -> Vector R -> Vector R
forwards t v = foldr forward v $ reverse t

forward :: Matrix R -> Vector R -> Vector R
forward m v = sigmoid $ m #> v

sigmoid n = 1 / (1 + exp (-n))

someFunc :: IO ()
someFunc = putStrLn "someFunc"
