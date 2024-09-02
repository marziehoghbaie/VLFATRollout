import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from pretty_confusion_matrix import pp_matrix
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve


def testROC(loader, model, y_train, loss_fn, logger, phase, device='cuda', save_path=None):
    model.eval()
    running_corrects = 0
    running_loss = 0
    trues = []
    y_score = []
    y_pred = []

    all_sample_name = []
    all_bscan_ids = []

    with torch.no_grad():
        for idx, (images, imgs_view, bscan_ids, labels, true_label, sample_names) in enumerate(loader):

            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs_batch = F.softmax(logits, dim=1).data
            prob, preds = torch.max(probs_batch, dim=1)
            trues.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if len(probs_batch[0]) > 2:
                probs_batch = probs_batch.cpu().numpy()
            else:
                probs_batch = [probs_batch[0][1].cpu().numpy()]
            y_score.extend(probs_batch)
            all_sample_name.extend(sample_names)
            name = ', '.join(str(i.cpu().numpy()[0]) for i in bscan_ids)
            all_bscan_ids.append(name)

            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    accuracy = running_corrects / len(loader.dataset)
    'y_true, y_pred'
    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info('[INFO] {} acc, balanced accuracy,  and loss: {}, {}, {}'.format(phase, accuracy, balanced_acc, loss))

    df_results = pd.DataFrame(data={
        'FilesetID': all_sample_name,
        'BscanIDs': all_bscan_ids,
        'true_label': trues,
        'prediction': y_pred,
        'y_score': y_score})
    df_results.to_csv(f'{save_path}/results.csv')

    classification_report = metrics.classification_report(y_true=trues, y_pred=y_pred,
                                                          target_names=loader.dataset.categories)
    logger.info('[INFO] classification report \n')
    logger.info(classification_report)

    n_classes = len(loader.dataset.categories)
    categories = loader.dataset.categories
    n_test = 1
    draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories)
    micro_roc_auc_ovr, aps = calculate_roc(y_score=y_score, y_test=trues, y_train=y_train, logger=logger)

    return accuracy, loss, balanced_acc, micro_roc_auc_ovr, aps


def calculate_roc(y_score, y_test, y_train, logger):
    """in a multi class high imbalanced setting, micro-averaging is preferable over macro-averaging"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    micro_roc_auc_ovr = roc_auc_score(
        y_onehot_test,
        y_score,
        multi_class="ovr",
        average="macro")

    aps = average_precision_score(y_onehot_test, y_score, average='weighted')

    logger.info(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")
    logger.info(f"average_precision_score:\n{aps:.2f}")
    return micro_roc_auc_ovr, aps


def draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories):
    # # draw the confusion matrix
    conf_matrix = metrics.confusion_matrix(trues, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=range(1, n_classes + 1),
                         columns=categories)

    cmap = 'OrRd'
    pp_matrix(df_cm, cmap=cmap)
    plt.savefig('{}/conf_mtrx_{}.png'.format(save_path, str(n_test)))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=categories)

    cm_display.plot(cmap="OrRd", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('{}/conf_mtrx_simple_{}.png'.format(save_path, str(n_test)))


def draw_roc(y_true, y_prob, save_path, logger):
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    AP = average_precision_score(y_true, y_prob, average="macro")
    display = PrecisionRecallDisplay(
        recall=recall,
        precision=precision,
        average_precision=AP,
    )
    display.plot()
    _ = display.ax_.set_title(f"Precision Recall curve(full volumes)")
    plt.savefig('{}/PrecisionRecall_curve.png'.format(save_path))

    RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_prob,
        name=f"ROC Curve",
        color="darkorange")

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.savefig('{}/ROC.png'.format(save_path))
    auc = roc_auc_score(y_true, y_prob)
    logger.info(f'[INFO] AUC score: {auc}')
    return auc

