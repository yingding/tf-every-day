from datetime import datetime
from pytz import timezone as ptimezone
import tensorflow
# from tensorflow.keras.callbacks import ProgbarLogger, Callback

def get_local_time_str(target_tz_str: str = "Europe/Berlin", format_str: str = "%Y-%m-%d %H-%M-%S") -> str:
    """
    this method is created since the local timezone is miss configured on the server
    @param: target timezone str default "Europe/Berlin"
    @param: "%Y-%m-%d %H-%M-%S" returns 2022-07-07 12-08-45
    """
    target_tz = ptimezone(target_tz_str) # create timezone, in python3.9 use standard lib ZoneInfo
    # utc_dt = datetime.now(datetime.timezone.utc)
    target_dt = datetime.now(target_tz)
    return datetime.strftime(target_dt, format_str)


class MultiEpochProgbarLogger(tensorflow.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode='samples', stateful_metrics=None, display_per_epoch=1000, verbose=1):
        super().__init__(count_mode, stateful_metrics)
        self.display_per_epochs = display_per_epoch
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.display_per_epochs == 0:
            super().on_epoch_end(epoch, logs)


# class NBatchLogger(Callback):
#     """
#     A Logger that log average performance per `display` steps.
#     """
#     def __init__(self, display):
#         self.step = 0
#         self.display = display
#         self.metric_cache = {}

#     def on_batch_end(self, batch, logs={}):
#         self.step += 1
#         for k in self.params['metrics']:
#             if k in logs:
#                 self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
#         if self.step % self.display == 0:
#             metrics_log = ''
#             for (k, v) in self.metric_cache.items():
#                 val = v / self.display
#                 if abs(val) > 1e-3:
#                     metrics_log += ' - %s: %.4f' % (k, val)
#                 else:
#                     metrics_log += ' - %s: %.4e' % (k, val)
#             print('step: {}/{} ... {}'.format(self.step,
#                                           self.params['steps'],
#                                           metrics_log))
#             self.metric_cache.clear()


# class NBatchProgBarLogger(ProgbarLogger):
#     """
#     https://github.com/keras-team/keras/issues/2850#issuecomment-497987989
#     """
#     def __init__(self, count_mode='samples', stateful_metrics=None, display_per_batches=1000, verbose=1):
#         # super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
#         super().__init__(count_mode, stateful_metrics)
#         self.display_per_batches = display_per_batches
#         self.display_step = 1
#         self.verbose = verbose
    
#     @staticmethod
#     def dump(obj):
#         for attr in dir(obj):
#             print("obj.%s = %r" % (attr, getattr(obj, attr)))    

#     def on_train_begin(self, logs=None):
#          # self.dump(self)
#          self.epochs = self.params['epochs']


#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}
#         batch_size = logs.get('size', 0)
#         # In case of distribution strategy we can potentially run multiple steps
#         # at the same time, we should account for that in the `seen` calculation.
#         num_steps = logs.get('num_steps', 1)
#         if self.use_steps:
#             self.seen += num_steps
#         else:
#             self.seen += batch_size * num_steps

#         for k in self.params['metrics']:
#             if k in logs:
#                 self.log_values.append((k, logs[k]))

#         self.display_step += 1
#         # Skip progbar update for the last batch;
#         # will be handled by on_epoch_end.
#         if self.verbose and self.seen < self.target and self.display_step % self.display_per_batches == 0:
#             self.progbar.update(self.seen, self.log_values) 