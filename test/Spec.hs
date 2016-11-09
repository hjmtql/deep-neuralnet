import Test.Hspec
import Test.QuickCheck
import Control.Exception (evaluate)

import Lib
import Numeric.LinearAlgebra
import Control.Monad

main :: IO ()
main = hspec .
  describe "Prelude.head" $ do
    it "returns the first element of a list" $
      head [23 ..] `shouldBe` (23 :: Int)

    it "returns the first element of an *arbitrary* list" .
      property $ \x xs -> head (x:xs) == (x :: Int)

    it "throws an exception if used with an empty list" $
      evaluate (head []) `shouldThrow` anyException

    it "size is ok?: forward" $ do
      let
        w1 = matrix 3 (replicate 9 1)
        w2 = matrix 4 (replicate 4 1)
        u = forward w1 $ matrix 3 (replicate 6 1)
      size (forward w2 u) `shouldBe` size (matrix 3 (replicate 3 1))

    it "size is ok?: forwardU" $ do
      let
        w1 = matrix 3 (replicate 9 1)
        w2 = matrix 4 (replicate 4 1)
        u = w1 <> inputWithBias is
        is = matrix 3 (replicate 6 1)
      size (forwardU w2 u) `shouldBe` size (matrix 3 (replicate 3 1))

    it "size is ok?: forwardUs" $ do
      ws <- genTensor [2, 3, 1]
      let
        u = head ws <> inputWithBias is
        is = matrix 3 (replicate 6 1)
      (size . last) (forwardUs (tail ws) u) `shouldBe` size (matrix 3 (replicate 3 1))

    it "size is ok?: backPropagation" $ do
      ws <- genTensor [2, 3, 1]
      let
        i = matrix 3 (replicate 6 1)
        o = matrix 3 (replicate 3 1)
      zipWithM_ shouldBe (fmap size (backPropagation i o ws)) (fmap size ws)
