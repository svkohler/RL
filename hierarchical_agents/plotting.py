import matplotlib.pyplot as plt
import time
import numpy as np

from helper import relative_delta

class Plot_world():
    def __init__(self, world):
        """sets up the plot
        """
        self.world = world
        plt.ion()
        plt.axes().set_aspect('equal')
        self.redraw()

    def redraw(self): 
        plt.clf()

        plt.plot(self.world.goal[0], self.world.goal[1], marker="o", c="gold",  markersize=20)         
        for wall in self.world.walls: 
            ((x0,y0),(x1,y1)) = wall 
            plt.plot([x0,x1],[y0,y1],"-k",linewidth=3)
    
    def show_and_close(self, duration=5):
        """Shows the plot for a specified duration (in seconds) and then closes it."""
        plt.draw()  # Ensure the plot is updated
        plt.pause(0.001)  # Small pause to render the plot
        time.sleep(duration)  # Wait for the specified duration
        plt.close()  # Close the plot

class Plot_env():
    def __init__(self, world, body=None,top=None):
        """sets up the plot
        """
        self.world = world
        self.body = body
        self.top = top
        plt.ion()
        plt.axes().set_aspect('equal')
        self.redraw()

    def redraw(self): 
        plt.clf()

        plt.plot(self.world.goal[0], self.world.goal[1], marker="o", c="gold",  markersize=20)        
        plt.plot(self.body.return_state()["rob_x_pos"], self.body.return_state()["rob_y_pos"], marker="X", c="blue", markersize=20)        
        plt.plot(self.body.route[0][0], self.body.route[0][1], marker="X", c="green", markersize=20)        
        for wall in self.world.walls: 
            ((x0,y0),(x1,y1)) = wall 
            plt.plot([x0,x1],[y0,y1],"-k",linewidth=3)

        if self.body.route or self.body.wall_history:
            self.plot_run()

    def plot_run(self):
        """plots the history after the agent has finished. This is typically only used if body.plotting==False """
        if self.body.route:
            for pos in self.body.route:
                plt.plot(pos[0],pos[1],"go", alpha=0.3)
                for whisker in self.body.whisker_set.set:
                    w = tuple(map(sum, zip((pos[0], pos[1]), (relative_delta(pos[2] + whisker.relative_dir, whisker.length)))))
                    plt.plot([pos[0], w[0]],[pos[1], w[1]], color="grey", alpha=0.3)
                plt.draw()
                plt.pause(0.001)
    
    def show_and_close(self, duration=5):
        """Shows the plot for a specified duration (in seconds) and then closes it."""
        plt.draw()  # Ensure the plot is updated
        plt.pause(0.001)  # Small pause to render the plot
        time.sleep(duration)  # Wait for the specified duration
        plt.close()  # Close the plot

class Plot_metric():
    def __init__(self, loss_lists, y_labels=[""], x_labels=[""], titles=[""]):
        """sets up the plot
        """
        self.loss_lists = loss_lists
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.titles = titles
        _, self.axs = plt.subplots(nrows=int(np.ceil(len(self.loss_lists)/2)), ncols=2, figsize=(8, 4))
        plt.ion()
        # plt.axes().set_aspect('equal')
        self.redraw(loss_lists, y_labels, x_labels, titles)

    def redraw(self, loss_lists, y_labels=[""], x_labels=[""], titles=[""]): 
        # plt.clf()
        for i in range(len(loss_lists)):
            self.axs[i].clear()
            self.axs[i].plot(loss_lists[i], linestyle="-", marker=".")
            self.axs[i].set_xlabel(x_labels[i])
            self.axs[i].set_ylabel(y_labels[i])
            self.axs[i].set_title(titles[i])
        # plt.draw()
        plt.pause(.0000001)