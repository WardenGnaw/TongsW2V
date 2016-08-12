import Data
import Preprocessing

if __name__ == "__main__":
    train_pos, train_neg, test_pos, test_neg = Data.GetIMDBData()
    temp = Preprocessing.Preprocess(train_pos, PreprocessPOS=True, PreprocessStem=True,
    PreprocessStopword=True, PreprocessContractions=True,
    PreprocessEmoticons=True)
    print(temp[0])
