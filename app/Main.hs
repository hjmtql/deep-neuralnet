module Main where

import Numeric.LinearAlgebra
import Common
import Forward
import BackProp
import AutoEncoder
import ActivationFunction
import Other

main :: IO ()
main = do
  regression
  classification

regression :: IO ()
regression = do
  ws <- genWeights [2, 4, 8, 16, 1]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [0, 1, 1, 0]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  -- let nws = last . take 500 $ iterate (backPropRegression (sigmoid, dsigmoid) (x, y)) ws
  nws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropRegression (sigmoid, dsigmoid)) ws
  let pws = preTrains (sigmoid, dsigmoid) x ws -- TODO: iter parameter
  npws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropRegression (sigmoid, dsigmoid)) pws
  putStrLn "training inputs"
  print x
  putStrLn "training outputs"
  print y
  putStrLn "inputs"
  print i
  putStrLn "not trained outputs"
  print $ forwardRegressions sigmoid ws i
  putStrLn "trainined outputs"
  print $ forwardRegressions sigmoid nws i
  putStrLn "pretrainined outputs"
  print $ forwardRegressions sigmoid npws i

classification :: IO ()
classification = do
  ws <- genWeights [2, 4, 8, 16, 3]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [1, 0, 0, 0,
                    0, 1, 1, 0,
                    0, 0, 0, 1]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  -- let nws = last . take 500 $ iterate (backPropClassification (sigmoid, dsigmoid) (x, y)) ws
  nws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropClassification (sigmoid, dsigmoid)) ws
  let pws = preTrains (sigmoid, dsigmoid) x ws -- TODO: iter parameter
  npws <- last . take 500 $ iterateM (sgdMethod 2 (x, y) $ backPropClassification (sigmoid, dsigmoid)) pws
  putStrLn "training inputs"
  print x
  putStrLn "training outputs"
  print y
  putStrLn "inputs"
  print i
  putStrLn "not trained outputs"
  print $ forwardClassification sigmoid ws i
  putStrLn "trainined outputs"
  print $ forwardClassification sigmoid nws i
  putStrLn "pretrainined outputs"
  print $ forwardClassification sigmoid npws i
