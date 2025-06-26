from plotting import Plot_metric

class TrainMonitor():
    """
    class to monitor training progress
    """
    def __init__(self):
        self.num_episodes = 0
        self.total_steps = 0
        self.performance_metric_coll, self.performance_metric_coll_ma = [], []
        self.computation_metric_coll, self.computation_metric_coll_ma = [], []
        self.pmetric = Plot_metric(
            [self.performance_metric_coll_ma, self.computation_metric_coll_ma], 
            y_labels=["step reward", "time per episode"], 
            x_labels=["episodes", "episodes"], 
            titles=["avg. reward per step", "avg. time per episode"]
        )

        self.grace_period = 100
        self.ma_window = 1000

    def update(self, episode_reward, episode_steps, comp_time):
        self.num_episodes += 1
        self.total_steps += episode_steps

        self.performance_metric_coll.append(episode_reward / episode_steps)
        self.computation_metric_coll.append(comp_time)

        if self.num_episodes > self.grace_period:
            self.performance_metric_coll_ma.append(
                    (sum(self.performance_metric_coll[-self.ma_window:])) / min(len(self.performance_metric_coll), self.ma_window)
                )
            self.computation_metric_coll_ma.append(
                    (sum(self.computation_metric_coll[-self.ma_window:])) / min(len(self.computation_metric_coll), self.ma_window)
                )
        self.pmetric.redraw([self.performance_metric_coll_ma, self.computation_metric_coll_ma])