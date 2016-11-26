module Main where

import Numeric.LinearAlgebra
import Common
import Forward
import BackProp
import AutoEncoder
import ActivFunc
import Other
import Mnist

main :: IO ()
main = do
  m <- parseCsvToMatrixR "data/mnist_test_100.csv"
  let (y, x) = mnistRead m
  ws <- genWeights [784, 10]

  nws <- last . take 500 $ iterateM (sgdMethod 100 (x, y) $ backPropClassification 0.01 sigmoids) ws
  let pws = preTrains 0.1 500 sigmoids x ws
  npws <- last . take 500 $ iterateM (sgdMethod 100 (x, y) $ backPropClassification 0.01 sigmoids) pws

  putStrLn "not trained outputs"
  print $ tr $ forwardClassification sigmoidC ws x
  putStrLn "trainined outputs"
  print $ forwardClassification sigmoidC nws x
  putStrLn "pretrainined outputs"
  print $ forwardClassification sigmoidC npws x
