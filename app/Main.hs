module Main where

import Common
import Forward
import BackProp
import AutoEncoder
import Numeric.LinearAlgebra

main :: IO ()
main = do
  ws <- genTensor [2, 12, 8, 4, 1]
  let x = matrix 4 [0, 0, 1, 1, 0, 1, 0, 1]
  let y = matrix 4 [0, 1, 1, 0]
  let i = matrix 4 [0, 0, 1, 1, 0, 1, 0, 1] -- example
  let nws = foldr (\f x -> f x) ws (replicate 2000 (backPropagation x y))
  let pws = preTrains x ws
  let npws = foldr (\f x -> f x) pws (replicate 2000 (backPropagation x y))
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
  putStrLn "pretrainined outputs"
  print $ forwards npws i
