module Main where

import Numeric.LinearAlgebra
import Common
import Forward
import BackProp
import AutoEncoder
import ActivFunc
import Other

main :: IO ()
main = do
  regression
  classification

regression :: IO ()
regression = do
  ws <- genWeights [2, 4, 8, 1]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [0, 1, 1, 0]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  -- let nws = last . take 500 $ iterate (backPropRegression sigmoids (x, y)) ws
  nws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropRegression 0.1 sigmoids) ws
  let pws = preTrains 0.1 500 sigmoids x ws
  npws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropRegression 0.1 sigmoids) pws
  putStrLn "training inputs"
  print x
  putStrLn "training outputs"
  print y
  putStrLn "inputs"
  print i
  putStrLn "not trained outputs"
  print $ forwardRegression sigmoidC ws i
  putStrLn "trainined outputs"
  print $ forwardRegression sigmoidC nws i
  putStrLn "pretrainined outputs"
  print $ forwardRegression sigmoidC npws i

classification :: IO ()
classification = do
  ws <- genWeights [2, 4, 8, 3]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [1, 0, 0, 0,
                    0, 1, 1, 0,
                    0, 0, 0, 1]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  -- let nws = last . take 500 $ iterate (backPropClassification sigmoids (x, y)) ws
  nws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropClassification 0.1 sigmoids) ws
  let pws = preTrains 0.1 500 sigmoids x ws
  npws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropClassification 0.1 sigmoids) pws
  putStrLn "training inputs"
  print x
  putStrLn "training outputs"
  print y
  putStrLn "inputs"
  print i
  putStrLn "not trained outputs"
  print $ forwardClassification sigmoidC ws i
  putStrLn "trainined outputs"
  print $ forwardClassification sigmoidC nws i
  putStrLn "pretrainined outputs"
  print $ forwardClassification sigmoidC npws i
