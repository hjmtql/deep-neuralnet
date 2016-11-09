-- module Lib
--     ( someFunc
--     ) where

module Lib where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

inputWithBias :: Matrix R -> Matrix R
inputWithBias vs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias = fromLists . fmap init . toLists

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (uncurry randn) ts
  where ts = drop 1 . zip ns $ 0 : fmap (+1) ns -- +1: bias

forwards :: Tensor -> Matrix R -> Matrix R
forwards ws vs = foldr forward vs $ reverse ws

forward :: Matrix R -> Matrix R -> Matrix R
forward w vs = sigmoid $ w <> vbs
  where vbs = inputWithBias vs

sigmoid n = 1 / (1 + exp (-n))
dsigmoid n = sigmoid n * (1 - sigmoid n)

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

someFunc :: IO ()
someFunc = do
  ws <- genTensor [2, 4, 1]
  let x = matrix 4 [0, 0, 1, 1, 0, 1, 0, 1]
  let y = matrix 4 [0, 1, 1, 0]
  let i = matrix 4 [0, 0, 1, 1, 0, 1, 0, 1] -- example
  let nws = foldr (\f x -> f x) ws (replicate 10000 (backPropagation x y))
  putStrLn "training inputs"
  print x
  putStrLn "training outputs"
  print y
  putStrLn "inputs"
  print i
  putStrLn "not trained outputs"
  print $ forwards ws i
  putStrLn "trainined outputs"
  print $ forwards nws i
