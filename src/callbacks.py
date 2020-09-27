import numpy as np
from src.metrics import iou, accuracy, precision, recall


def get_accuracy_score(targets, predictions):
    assert len(targets) == len(predictions)
    met = []
    for i in range(len(targets)):
        met.append(accuracy(predictions[i], targets[i]).item())

    return np.mean(met)

def get_precision_score(targets, predictions):
    assert len(targets) == len(predictions)
    met = []
    for i in range(len(targets)):
        met.append(precision(predictions[i], targets[i]).item())

    return np.mean(met)

def get_recall_score(targets, predictions):
    assert len(targets) == len(predictions)
    met = []
    for i in range(len(targets)):
        met.append(recall(predictions[i], targets[i]).item())

    return np.mean(met)

def get_iou_score(targets, predictions):
    assert len(targets) == len(predictions)
    met = []
    for i in range(len(targets)):
        met.append(iou(predictions[i], targets[i]).item())

    return np.mean(met)



#class to keep track of user selected metrcs during training
class TrainMetricRecorder:
    
    METRICS = {'accuracy': get_accuracy_score, 'precision': get_precision_score, 'recall': get_recall_score, 'iou': get_iou_score}
    
    def __init__(self, metrics):
        self.history = {}
        self.history['train_loss'] = []
        self.history['val_loss'] = []
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions =[]
        self.val_targets = []
        self.train_loss = []
        self.val_loss = []
        for metric in metrics:
            if metric not in self.METRICS:
                raise ValueError(f'{metric} is not a valid metric. Valid metrics are: {TrainMetricRecorder.METRICS}')
            else:
                self.history['train_'+metric] = []
                self.history['val_'+metric] = []
        self.metrics = metrics        

    def on_train_batch_end(self, y_true, y_preds, loss):
        self.train_targets.append(y_true)
        self.train_predictions.append(y_preds)
        self.train_loss.append(loss)

    def on_val_batch_end(self, y_true, y_preds, loss):
        self.val_targets.append(y_true)
        self.val_predictions.append(y_preds)
        self.val_loss.append(loss)                

    def on_epoch_start(self):
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions =[]
        self.val_targets = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self):
        for metric_name in self.metrics:
            if metric_name != 'accuracy':
                if len(self.train_predictions) > 0:
                    self.history['train_'+metric_name].append(self.METRICS[metric_name](self.train_targets, self.train_predictions))

                if len(self.val_predictions)>0:
                    self.history['val_'+metric_name].append(self.METRICS[metric_name](self.val_targets, self.val_predictions))
            else:
                if len(self.train_predictions) > 0:
                    self.history['train_'+metric_name].append(self.METRICS[metric_name](self.train_targets, self.train_predictions))

                if len(self.val_predictions) > 0:    
                    self.history['val_'+metric_name].append(self.METRICS[metric_name](self.val_targets, self.val_predictions))

        #calculate average loss
        if len(self.train_loss) > 0:
            self.history['train_loss'].append(sum(self.train_loss)/len(self.train_loss))

        if len(self.val_loss) > 0:
            self.history['val_loss'].append(sum(self.val_loss)/len(self.val_loss))    

