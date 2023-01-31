import random

import libmr
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spd
from keras.datasets import cifar10, mnist
from keras.models import Model
from keras.models import load_model
from skimage.color import rgb2gray
from skimage.transform import resize

dataset = mnist
eu_weight = 1e-28
cos_weight = 1e-10
tail = 500
test_size = 2500


def calculate_logits_and_predictions(model: Model, x_train, classes):
    model_without_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)
    logits = model_without_softmax.predict(x_train)
    sofrmaxes = np.array([softmax(lg) for lg in logits])
    predictions = np.array([classes[s] for s in np.argmax(sofrmaxes, axis=1)])
    return logits, predictions


def run_test(model, classes, weinbull_models, x_test, y_test):
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    logits = model.predict(x_test)
    openmax_probabs = np.array([recalibrate_scores(weinbull_models, lg, classes, alpharank=len(classes)) for lg in logits])
    softmax_prob = np.array([softmax(lg) for lg in logits])
    softmax_cl = np.argmax(softmax_prob, axis=-1)
    openmax_cl = np.argmax(openmax_probabs, axis=-1)
    correct_classifications = 0
    wrong_classifications = 0
    correct_rejections = 0
    wrong_rejections = 0

    for y, sc, oc in zip(y_test, softmax_cl, openmax_cl):
        if y == sc == oc:
            correct_classifications += 1
        elif oc == len(classes):
            if y == -1:
                correct_rejections += 1
            else:
                wrong_rejections += 1
        else:
            wrong_classifications += 1
    print(f"{correct_rejections=}")
    print(f"{wrong_rejections=}")
    print(f"{correct_classifications=}")
    print(f"{wrong_classifications=}")
    print(f"of {len(x_test)} examples")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def fit_weibull(activations, predictions, true_labels, classes, taillength):
    correct = np.where(predictions == true_labels)[0]
    weibull_model = {}
    for cl in classes:
        i = np.where(predictions[correct] == cl)[0]
        act = activations[i, :]
        mean_act = np.mean(act, axis=0)
        # Compute all, for this class, correctly classified images' distance to the MAV.
        tailtofit = sorted(
            [spd.euclidean(mean_act, act[col, :]) * eu_weight + cos_weight*spd.cosine(mean_act, act[col, :])
             for col in range(len(act))]
        )[-taillength:]
        print(tailtofit)
        print(f"{mean_act=}")
        weibull_model[cl] = {}
        weibull_model[cl]['mav'] = mean_act
        mr = libmr.MR(verbose=True)
        mr.fit_high(tailtofit, taillength)
        weibull_model[cl]['model'] = mr

    return weibull_model


def recalibrate_scores(weibull_model, img_layer_act, classes, alpharank=10):
    num_labels = len(classes)
    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]
    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_labels)
    for i, a in enumerate(alpha_weights):
        ranked_alpha[ranked_list[i]] = a
    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in classes:
        label_weibull = weibull_model[categoryid]['model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[categoryid]['mav']  # Obtain MAV for specific class.

        img_dist = spd.euclidean(label_mav, img_layer_act) * eu_weight + cos_weight * spd.cosine(label_mav, img_layer_act)
        weibull_score = label_weibull.w_score(img_dist)
        modified_layer_act = img_layer_act[categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [img_layer_act[categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)
    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit, classes)
    return openmax_probab


def compute_open_max_probability(openmax_known_score, openmax_unknown_score, classes):
    """
    Compute the OpenMax probability.
    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """
    prob_closed, scores = [], [],

    # Compute denominator for closet set + open set normalization.
    # Sum up the class scores.
    for category in classes:
        scores.append(np.exp(openmax_known_score[category]))
    scores = np.array(scores)
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)
    # Scores for image belonging to either closed or open set.
    prob_closed = scores / total_denominator
    prob_open = np.exp(openmax_unknown_score) / total_denominator
    probs = np.append(prob_closed, prob_open)
    assert len(probs) == len(classes) + 1
    return probs


def main():
    animal_labels = np.array([2, 3, 4, 5, 6])
    vehicle_labels = np.array([0, 1, 7, 8, 9])

    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()
    labels = vehicle_labels

    y_train_all, y_test_all = y_train_all.flatten(), y_test_all.flatten()
    x_train = x_train_all[np.isin(y_train_all, labels, ).ravel()]/255
    y_train = y_train_all[np.isin(y_train_all, labels, ).ravel()]
    x_test = x_test_all[np.isin(y_test_all, labels, ).ravel()]/255
    y_test = y_test_all[np.isin(y_test_all, labels, ).ravel()]

    # x_train = np.expand_dims(x_train, axis=-1)
    # x_test = np.expand_dims(x_test, axis=-1)

    classes = list(range(len(animal_labels)))
    print(classes)
    for i, v in enumerate(labels):
        y_train[y_train == v] = i
    for i, v in enumerate(labels):
        y_test[y_test == v] = i

    K = len(classes)
    model = load_model("cifar-vehicles")
    logits, predictions = calculate_logits_and_predictions(model, x_train, classes)

    weinbull_models = fit_weibull(logits, predictions, y_train, classes, taillength=tail)
    x_f = x_test_all[np.isin(y_test_all, animal_labels, ).ravel()]/255
    # x_f = x_f[np.random.randint(len(x_f), size=test_size)]
    print("\nActual: ")
    run_test(model, classes, weinbull_models, x_test, y_test)
    print("\nFooling: ")
    run_test(model, classes, weinbull_models, x_f, np.array([-1] * len(x_f)))


if __name__ == '__main__':
    main()
