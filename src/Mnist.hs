module Mnist (
    mnistRead,
    classesToLabels
  ) where

import Numeric.LinearAlgebra

mnistRead :: Int -> Matrix R -> (Matrix R, Matrix R)
mnistRead i m = (labelsToClasses i l, cmap (/ 255) x)
  where (l, x) = separateLabelAndFeature m

separateLabelAndFeature :: Matrix R -> (Matrix R, Matrix R)
separateLabelAndFeature m = (takeColumns 1 m, dropColumns 1 m)

labelsToClasses :: Int -> Matrix R -> Matrix R
labelsToClasses i = fromLists . fmap (labelToClass i . head) . toLists

labelToClass :: Int -> R -> [R]
labelToClass i l = fmap (\j -> if j == truncate l then 1 else 0) [0..i-1]

classesToLabels :: Matrix R -> Matrix R
classesToLabels = fromLists . fmap ((:[]) . classToLabel) . toLists

classToLabel :: [R] -> R
classToLabel = fromIntegral . maxIndex . vector
