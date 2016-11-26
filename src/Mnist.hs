module Mnist (
    mnistRead
  ) where

import Numeric.LinearAlgebra

mnistRead :: Matrix R -> (Matrix R, Matrix R)
mnistRead m = (tr $ mnistLabelsToClasses l, tr $ cmap (/ 255) x)
  where (l, x) = separateLabelAndFeature m

separateLabelAndFeature :: Matrix R -> (Matrix R, Matrix R)
separateLabelAndFeature m = (takeColumns 1 m, dropColumns 1 m)

mnistLabelsToClasses :: Matrix R -> Matrix R
mnistLabelsToClasses = fromLists . fmap (mnistLabelToClass . head) . toLists

mnistLabelToClass :: R -> [R]
mnistLabelToClass l = case l of
  0 -> [1,0,0,0,0,0,0,0,0,0]
  1 -> [0,1,0,0,0,0,0,0,0,0]
  2 -> [0,0,1,0,0,0,0,0,0,0]
  3 -> [0,0,0,1,0,0,0,0,0,0]
  4 -> [0,0,0,0,1,0,0,0,0,0]
  5 -> [0,0,0,0,0,1,0,0,0,0]
  6 -> [0,0,0,0,0,0,1,0,0,0]
  7 -> [0,0,0,0,0,0,0,1,0,0]
  8 -> [0,0,0,0,0,0,0,0,1,0]
  9 -> [0,0,0,0,0,0,0,0,0,1]
  _ -> []
