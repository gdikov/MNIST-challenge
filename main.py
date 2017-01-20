import time

from experiments.evaluate_models import test_knn, test_logreg, test_convnet, test_gp

def measure_test_time(fn):
    start_timer = time.time()
    result = fn()
    return (time.time() - start_timer, result)

if __name__ == "__main__":
    model_names = ("k-NN", )
    model_testers = (test_logreg,)# test_logreg)#, test_convnet, test_gp)

    for name, test in zip(model_names, model_testers):
        print("Evaluating the {} classifier...".format(name))
        test_time, result = measure_test_time(test)
        print("\tEvaluated in {} s (including data loading time)".format(test_time))
        print("\tTest accuracy = {0}% (Test error = {1}%)".format(result, 100. - result))
