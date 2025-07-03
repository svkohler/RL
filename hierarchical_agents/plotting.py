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
                plt.close()
    
    def show_and_close(self, duration=5):
        """Shows the plot for a specified duration (in seconds) and then closes it."""
        plt.draw()  # Ensure the plot is updated
        plt.pause(0.001)  # Small pause to render the plot
        time.sleep(duration)  # Wait for the specified duration
        plt.close()  # Close the plot

class Plot_metric:
    def __init__(self, loss_lists, y_labels=None, x_labels=None, titles=None, figsize=(8, 4)):
        """
        Initialize a dynamic plot in a single figure with multiple subplots.
        """
        # Default values for labels and titles
        y_labels = y_labels if y_labels and len(y_labels) == len(loss_lists) else [""] * len(loss_lists)
        x_labels = x_labels if x_labels and len(x_labels) == len(loss_lists) else [""] * len(loss_lists)
        titles = titles if titles and len(titles) == len(loss_lists) else [""] * len(loss_lists)

        # Create subplots
        nrows = int(np.ceil(len(loss_lists) / 2))  # Number of rows
        ncols = 2  # Number of columns
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.ax = self.ax.ravel() if isinstance(self.ax, np.ndarray) else [self.ax]  # Flatten to 1D array

        # Initialize lines and data
        self.lines = {}
        self.x_data = {}
        self.y_data = {}

        for i in range(len(loss_lists)):
            self.ax[i].set_title(titles[i])
            self.ax[i].set_xlabel(x_labels[i])
            self.ax[i].set_ylabel(y_labels[i])
            self.ax[i].grid(True)

            # Initialize line and data
            self.lines[i], = self.ax[i].plot(range(len(loss_lists[i])), loss_lists[i], label=f"Plot {i+1}")
            self.x_data[i] = list(range(len(loss_lists[i])))  # Ensure x_data is a list
            self.y_data[i] = list(loss_lists[i])  # Ensure y_data is a list

        # Hide unused subplots (if any)
        for j in range(len(loss_lists), len(self.ax)):
            self.ax[j].axis("off")

    def update(self, ys):
        """
        Update the data for the plot and redraw it.
        
        Args:
            x (list of arrays): List of x data for each subplot.
            y (list of arrays): List of y data for each subplot.
        """
        for i, _ in enumerate(self.lines):
            self.x_data[i] = range(len(ys[i]))  # Ensure x_data is a list
            self.y_data[i] = ys[i]  # Ensure y_data is a list
            self.lines[i].set_data(self.x_data[i], self.y_data[i])
            self.ax[i].relim()  # Recalculate limits
            self.ax[i].autoscale_view()  # Autoscale the view
        self.fig.canvas.draw()  # Redraw the figure
        self.fig.canvas.flush_events()  # Flush GUI events
