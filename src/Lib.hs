-- module Lib
--     ( someFunc
--     ) where

module Lib where

import Numeric.LinearAlgebra

type Tensor = [Matrix R]
type Layer = [Int]

-- input only
extendForBias :: Matrix R -> Matrix R
extendForBias vs = fromLists $ toLists vs `mappend` [replicate (cols vs) 1] -- [1,1,..1]: bias

-- weignt only
shrinkForBias :: Matrix R -> Matrix R
shrinkForBias = fromLists . fmap init . toLists

genTensor :: Layer -> IO Tensor
genTensor ns = mapM (\(x, y) -> uncurry randn (x, y + 1)) ts -- 1: bias
  where ts = drop 1 $ zip ns (0:ns)

forwards :: Tensor -> Matrix R -> Matrix R
forwards ws vs = foldr forward vs $ reverse ws

forward :: Matrix R -> Matrix R -> Matrix R
forward w vs = sigmoid $ w <> vbs
  where vbs = extendForBias vs

sigmoid n = 1 / (1 + exp (-n))
dsigmoid n = sigmoid n * (1 - sigmoid n)


-- TODO: simplify
backPropagation :: Matrix R -> Matrix R -> Tensor -> Tensor
backPropagation xs ys ws = zipWith (-) ws (fmap (* 0.1) dws)
  where
    dws = (/ len) <$> zipWith (<>) dels (fmap (tr . extendForBias) (init zs))
    dels = reverse $ calcDel (reverse (zip (init us) (tail ws))) del -- length: L - 1
    del = last zs - ys
    us = forwards2 (tail ws) u -- length: L - 1
    u = head ws <> extendForBias xs
    zs = xs : fmap sigmoid us -- length: L
    len = fromIntegral $ cols xs

forwards2 :: Tensor -> Matrix R -> Tensor
forwards2 ws u = [u] `mappend` case ws of
  [] -> []
  (m:ms) -> forwards2 ms nu
    where nu = forward2 m u

forward2 :: Matrix R -> Matrix R -> Matrix R
forward2 w u = w <> vs
  where vs = extendForBias $ sigmoid u

calcDel :: [(Matrix R, Matrix R)] -> Matrix R -> Tensor
calcDel uws d = [d] `mappend` case uws of
  [] -> []
  (t:ts) -> calcDel ts nd
    where
      nd = dsigmoid u * (tr (shrinkForBias w) <> d)
      (u, w) = t


someFunc :: IO ()
someFunc = do
  ws <- genTensor [2, 4, 1]
  let is = matrix 4 [0, 0, 1, 1, 0, 1, 0, 1]
  let os = matrix 4 [0, 1, 1, 0]
  let nws = foldr (\f x -> f x) ws (replicate 10000 (backPropagation is os))
  print $ forwards ws is
  print $ forwards nws is
  putStrLn "someFunc"
