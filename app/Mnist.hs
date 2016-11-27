module Main where

import Numeric.LinearAlgebra
import Control.Arrow
import Common
import Forward
import BackProp
import AutoEncoder
import ActivFunc
import Other
import Mnist

main :: IO ()
main = do
  tm <- parseCsvToMatrixR "data/mnist_test_100.csv"
  let (ty, tx) = tr *** tr $ mnistRead 10 tm
  m <- parseCsvToMatrixR "data/mnist_train_600.csv"
  let (y, x) = tr *** tr $ mnistRead 10 m
  ws <- genWeights [784, 10]

  -- let pws = preTrains 0.5 1000 ramps x ws
  nws <- last . take 1000 $ iterateM (sgdMethod 100 (x, y) $ backPropClassification 0.5 ramps) ws

  -- putStrLn "not trained outputs"
  -- print . tr . classesToLabels . tr . forwardClassification rampC ws $ tx
  -- TODO: use function
  let o = tr . classesToLabels . tr . forwardClassification rampC nws $ tx
  let t = tr . classesToLabels . tr $ ty
  let l = head . toLists $ t - o
  let a = length l
  let s = length . filter (==0) $ l
  print $ fromIntegral s / fromIntegral a

  -- putStrLn "pretrainined outputs"
  -- print . classesToLabels . tr . forwardClassification rampC npws $ x
