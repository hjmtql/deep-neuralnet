module Common (
    genWeights,
    initWeights,
    inputWithBias,
    weightWithoutBias,
    pickupSets
  ) where

import Numeric.LinearAlgebra
import System.Random.Shuffle

genWeights :: [Int] -> IO [Matrix R]
genWeights ns = mapM genWeight ts
  where ts = zip ns (tail ns)

genWeight :: (Int, Int) -> IO (Matrix R)
genWeight (i, o) = randn o (i + 1) -- +1: bias

initWeights :: [Int] -> [Matrix R]
initWeights ns = fmap initWeight ts
  where ts = zip ns (tail ns)

initWeight :: (Int, Int) -> Matrix R
initWeight (i, o) = matrix (i + 1) . replicate (o * (i + 1)) $ 0 -- +1: bias

inputWithBias :: Matrix R -> Matrix R
inputWithBias v = v === konst 1 (1, cols v) -- [1,1,..1]: bias

weightWithoutBias :: Matrix R -> Matrix R
weightWithoutBias w = w ?? (All, DropLast 1)

pickupSets :: Int -> (Matrix R, Matrix R) -> IO (Matrix R, Matrix R)
pickupSets n (x, y) = do
  s <- shuffleM [0..cols y - 1]
  let ps = take n s in
    return (x ¿ ps, y ¿ ps)
