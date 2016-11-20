module Other (
    parseCsvToMatrixR,
    printCsvFromMatrixR,
    iterateM
  ) where

import Numeric.LinearAlgebra
import Data.List
import Data.List.Split

parseCsvToMatrixR :: FilePath -> IO (Matrix R)
parseCsvToMatrixR fp = do
  csv <- readFile fp
  return . fromLists . fmap (fmap (read :: String -> R) . splitOn ",") $ lines csv

printCsvFromMatrixR :: FilePath -> Matrix R -> IO ()
printCsvFromMatrixR fp m = do
  let csv = unlines . fmap (intercalate ", ") $ fmap show <$> toLists m
  writeFile fp csv

-- TODO: check monad rule
iterateM :: Monad m => (a -> m a) -> a -> [m a]
iterateM f x = iterate (f =<<) (return x)
