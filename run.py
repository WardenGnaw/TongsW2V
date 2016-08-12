import sys
import argparse
import logging
from tqdm import tqdm

import Data
import Preprocessing
import Word2Vec
import Classifiers

def ParseArguments():
    """

    """
    parser = argparse.ArgumentParser()

    # Type
    type_group = parser.add_argument_group('type')
    # type_group.add_argument('-lexicon', action='store_true', default=False, help='Do sentiment analysis via Lexicon')
    # type_group.add_argument('-ML', action='store_true', default=False, help='Do sentiment analysis via Machine Learning')
    type_group.add_argument('-Word2Vec', action='store_true', default=False, help='Do sentiment analysis via Word2Vec')
    type_group.add_argument('-Word2VecPMI', action='store_true', default=False, help='Do sentiment analysis via Word2Vec with PMI')

    
    # Training Dataset
    train_dataset_group_title = parser.add_argument_group(title='Training Dataset', description='Dataset to use when training Word2Vec and classifier')
    train_dataset_group = train_dataset_group_title.add_mutually_exclusive_group(required=True)
    train_dataset_group.add_argument('-train_imdb', action='store_true', default=False, help='Use the IMDB dataset')
    train_dataset_group.add_argument('-train_twitter', action='store_true', default=False, help='Use the twitter dataset')
    train_dataset_group.add_argument('-train_file', action='store', default=False, help='Use file given by user.')
    train_dataset_group.add_argument('-skip_train', action='store_true', default=False, help='Skip training.')

    # Preprocessing flags
    preprocessing_group = parser.add_argument_group(title='preprocessing', description='What to do when preprocessing text')
    preprocessing_group.add_argument('-POS_tag', action='store_true', default=False, help='Enable flag to turn on POS tagging for preprocessing')
    preprocessing_group.add_argument('-stopword_removal', action='store_true', default=False, help='Enable flag to remove stopwords during preprocessing')
    preprocessing_group.add_argument('-stem', action='store_true', default=False, help='Enable flag to stem during preprocessing')
    preprocessing_group.add_argument('-expand_contractions', action='store_true', default=False, help='Enable flag to expand contractions during preprocessing')
    preprocessing_group.add_argument('-replace_emo', action='store_true', default=False, help='Enable flag to replace emotions to words during preprocessing')
    preprocessing_group.add_argument('-tag_neg', action='store_true', default=False, help='Enable flag to tag negative words during preprocessing')

    # Word2Vec
    word2vec_group = parser.add_argument_group('Word2Vec')
    word2vec_group.add_argument('-load_w2v', action='store', metavar='<w2v_name>')
    word2vec_group.add_argument('-use_google_vec', action='store_true')

    # Classifiers
    classifier_group_title = parser.add_argument_group(title='classifier', description='Classifier to use when classifying text')
    classifier_group_title.add_argument('-load_clf', action='store', metavar='<classifier_name>')
    classifier_group = classifier_group_title.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('-SVM', action='store_true', default=False, help='Enable flag to use Support Vector Machine for classification')
    classifier_group.add_argument('-RF', action='store_true', default=False, help='Enable flag to use Random Forest for classification')
    classifier_group.add_argument('-NB', action='store_true', default=False, help='Enable flag to use Naive Bayes for classification')
    classifier_group.add_argument('-LR', action='store_true', default=False, help='Enable flag to use Logistic Regression (aka logit, MaxEnt) for classification')
    
    # Test Dataset
    test_dataset_group_title = parser.add_argument_group(title='Test Dataset', description='Dataset to use when testing')
    test_dataset_group = test_dataset_group_title.add_mutually_exclusive_group(required=True)
    test_dataset_group.add_argument('-test_imdb', action='store_true', default=False, help='Use the IMDB dataset')
    test_dataset_group.add_argument('-test_twitter', action='store_true', default=False, help='Use the twitter dataset')
    test_dataset_group.add_argument('-test_file', action='store', default=False, help='Use file given by user.')
    test_dataset_group.add_argument('-test_dnc', action='store_true', default=False, help='Use dnc.')
    test_dataset_group.add_argument('-test_rnc', action='store_true', default=False, help='Use rnc.')
    test_dataset_group.add_argument('-no_test', action='store_true', default=False, help='Skip testing phase')

    # Check to see if it ran blank
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    results, unknown = parser.parse_known_args()
    return results, unknown

def Word2VecFunc(args, logger, pos_processed_data, neg_processed_data, unsup_processed_data):
    if not args.load_w2v and not args.use_google_vec:
        w2v_model = Word2Vec.TONGSWord2Vec(data=pos_processed_data + neg_processed_data + unsup_processed_data)

        w2v_model.Save('imdb.vec')
    elif args.use_google_vec:
        w2v_model = Word2Vec.TONGSWord2Vec()
    else:
        w2v_model = Word2Vec.TONGSWord2Vec(filename='imdb.vec')
    
    if not args.load_clf:
        clf = Classifiers.TONGSClassifier('imdb', SVM=args.SVM, RF=args.RF, NB=args.NB, LR=args.LR)

        pos_X = w2v_model.ConvertSentencesToVectors(pos_processed_data)
        neg_X = w2v_model.ConvertSentencesToVectors(neg_processed_data)

        X = pos_X + neg_X
        y = [1] * len(pos_X) + [0] * len(neg_X)

        clf.train(X, y)
        clf.save()
    else:
        clf = Classifiers.TONGSClassifier('imdb', rebuild=False, SVM=args.SVM, RF=args.RF, NB=args.NB, LR=args.LR)

    if args.test_dnc or args.test_rnc:
        data = Data.GetUnlabeledTestData(DNC=args.test_dnc, RNC=args.test_rnc)

        processed_data = Preprocessing.Preprocess(data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        X_data = w2v_model.ConvertSentencesToVectors(processed_data)

        pos = 0
        neg = 0

        for X in tqdm(X_data):
            if clf.classify(X) == 1:
                pos += 1
            elif clf.classify(X) == 0:
                neg += 1

        print('(pos, neg):', pos, neg)


    elif not args.no_test:
        pos_test_data, neg_test_data = Data.GetTestData(IMDB=args.test_imdb, twitter=args.test_twitter)

        logger.info('Processing positive test dataset')
        pos_processed_data = Preprocessing.Preprocess(pos_test_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        logger.info('Processing negative test dataset')
        neg_processed_data = Preprocessing.Preprocess(neg_test_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        pos_X = w2v_model.ConvertSentencesToVectors(pos_processed_data)
        neg_X = w2v_model.ConvertSentencesToVectors(neg_processed_data)

        correct_pos = 0

        for X in pos_X:
            if clf.classify(X) == 1:
                correct_pos += 1

        correct_neg = 0
        for X in neg_X:
            if clf.classify(X) == 0:
                correct_neg += 1
        

        print('Positive Accuracy:', (correct_pos) / len(pos_X))
        print('Negative Accuracy:', (correct_neg) / len(neg_X))
        print('Accuracy:', (correct_pos + correct_neg) / (len(pos_X) + len(neg_X)))

def Word2VecPMIFunc(args, logger, pos_processed_data, neg_processed_data, unsup_processed_data):
    if not args.load_w2v and not args.use_google_vec:
        w2v_model = Word2Vec.TONGSWord2Vec(data=pos_processed_data + neg_processed_data + unsup_processed_data)

        w2v_model.Save('imdb.vec')
    elif args.use_google_vec:
        w2v_model = Word2Vec.TONGSWord2Vec()
    else:
        w2v_model = Word2Vec.TONGSWord2Vec(filename='imdb.vec')
    
    if args.test_dnc or args.test_rnc:
        data = Data.GetUnlabeledTestData(DNC=test_dnc, RNC=test_rnc)

        processed_data = Preprocessing.Preprocess(pos_test_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )
        pos = 0
        neg = 0

        for X in tqdm(processed_data):
            if w2v_model.SentencePMI(X) > 0:
                pos += 1
            elif w2v_model.SentencePMI(X) < 0:
                neg += 1

        print('(pos, neg):', pos, neg)

    elif not args.no_test:
        pos_test_data, neg_test_data = Data.GetTestData(IMDB=args.test_imdb, twitter=args.test_twitter)

        logger.info('Processing positive test dataset')
        pos_processed_data = Preprocessing.Preprocess(pos_test_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        logger.info('Processing negative test dataset')
        neg_processed_data = Preprocessing.Preprocess(neg_test_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        correct_pos = 0

        for X in tqdm(pos_processed_data):
            if w2v_model.SentencePMI(X) > 0:
                correct_pos += 1

        correct_neg = 0
        for X in tqdm(neg_processed_data):
            if w2v_model.SentencePMI(X) < 0:
                correct_neg += 1

        print('Positive Accuracy:', (correct_pos) / len(pos_processed_data))
        print('Negative Accuracy:', (correct_neg) / len(neg_processed_data))
        print('Accuracy:', (correct_pos + correct_neg) / (len(pos_processed_data) + len(neg_processed_data)))



def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Word2Vec_Sentiment')

    args, unknown_args = ParseArguments()

    if not args.skip_train:
        logger.info('Getting training dataset')
        pos_data, neg_data, unsup_data = Data.GetTrainData(IMDB=args.train_imdb, twitter=args.train_twitter)

        logger.info('Processing positive training dataset')
        pos_processed_data = Preprocessing.Preprocess(pos_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        logger.info('Processing negative training dataset')
        neg_processed_data = Preprocessing.Preprocess(neg_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

        logger.info('Processing unsupervised training dataset')
        unsup_processed_data = Preprocessing.Preprocess(unsup_data,
                                 PreprocessPOS=args.POS_tag, 
                                 PreprocessStem=args.stem, 
                                 PreprocessStopword=args.stopword_removal,
                                 PreprocessContractions=args.expand_contractions,
                                 PreprocessEmoticons=args.replace_emo,
                                 PreprocessNegation=args.tag_neg
                                 )

    if args.Word2Vec:
        Word2VecFunc(args, logger, pos_processed_data, neg_processed_data, unsup_processed_data)
    elif args.Word2VecPMI:
        Word2VecPMIFunc(args, logger, pos_processed_data, neg_processed_data, unsup_processed_data)

    return 0

if __name__ == "__main__":
    sys.exit(main())
