import random
import sys

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
eu_weight = 0.000001
tail = 300
test_size = 1000

def calculate_logits_and_predictions(model: Model, x_train, classes):
    model_without_softmax = Model(inputs=model.input, outputs=model.layers[-2].output)
    logits = model_without_softmax.predict(x_train)
    sofrmaxes = np.array([softmax(lg) for lg in logits])
    predictions = np.array([classes[s] for s in np.argmax(sofrmaxes, axis=1)])
    return logits, predictions


def run_test(model, classes, weinbull_models, x_test, y_test):
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    logits = model.predict(x_test)
    openmax_probabs = np.array([recalibrate_scores(weinbull_models, lg, classes, alpharank=10) for lg in logits])
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
            [spd.euclidean(mean_act, act[col, :]) * eu_weight + spd.cosine(mean_act, act[col, :])
             for col in range(len(act))]
        )[-taillength:]
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

        img_dist = spd.euclidean(label_mav, img_layer_act) * eu_weight + spd.cosine(label_mav, img_layer_act)

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
    assert len(probs) == 11
    return probs


def main():
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    classes = np.unique(y_train)

    x_train = np.expand_dims(x_train, axis=-1) / 255
    x_test = np.expand_dims(x_test, axis=-1) / 255
    model = load_model("mnist")
    logits, predictions = calculate_logits_and_predictions(model, x_train, classes)

    weinbull_models = fit_weibull(logits, predictions, y_train, classes, taillength=tail)

    (_, _,), (x_f, y_f) = cifar10.load_data()
    x_f = np.expand_dims(
        np.array([resize(rgb2gray(im) / 255, (x_test[0].shape[0], x_test[0].shape[1]), anti_aliasing=True) for im in
                  x_f[:test_size]]),
        axis=-1)
    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    for i in range(4):
        ax[i].imshow(x_f[random.randint(0, test_size)], cmap='gray')
    plt.show()
    print("\nActual: ")
    run_test(model, classes, weinbull_models, x_test[:test_size], y_test[:test_size])
    print("\nFooling: ")
    run_test(model, classes, weinbull_models, x_f, np.array([-1] * test_size))


if __name__ == '__main__':
    main()
