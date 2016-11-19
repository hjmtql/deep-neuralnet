module Main where

import Common
import Forward
import BackProp
import AutoEncoder
import Numeric.LinearAlgebra
import ActivationFunction

main :: IO ()
main = do
  regression
  classification

regression :: IO ()
regression = do
  ws <- genWeights [2, 4, 1]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [0, 1, 1, 0]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  let nws = last . take 2000 $ iterate (backPropRegression sigmoid dsigmoid x y) ws
  let pws = preTrains sigmoid dsigmoid x ws
  let npws = last . take 2000 $ iterate (backPropRegression sigmoid dsigmoid x y) pws
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
  ws <- genWeights [2, 8, 3]
  let x = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1]
  let y = matrix 4 [1, 0, 0, 0,
                    0, 1, 1, 0,
                    0, 0, 0, 1]
  let i = matrix 4 [0, 0, 1, 1,
                    0, 1, 0, 1] -- example
  let nws = last . take 2000 $ iterate (backPropClassification sigmoid dsigmoid x y) ws
  let pws = preTrains sigmoid dsigmoid x ws
  let npws = last . take 2000 $ iterate (backPropClassification sigmoid dsigmoid x y) pws
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
