from experiments.evaluate_models import evaluate_knn, \
    evaluate_logreg, evaluate_convnet, evaluate_basicnet, evaluate_gp




if __name__ == "__main__":
    evaluate_knn(train_from_scratch=False)
    # evaluate_logreg(train_from_scratch=True)
    # evaluate_convnet(train_from_scratch=False, continue_from_checkpoint=False)
    # evaluate_gp(train_from_scratch=True, classification_mode='mixed_binary')
    # evaluate_basicnet(train_from_scratch=False)

