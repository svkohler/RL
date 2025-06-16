import matplotlib.pyplot as plt
import time

def main():
    # Example data points
    x_points = [1, 2, 3, 4, 5]
    y_points = [1, 4, 9, 16, 25]

    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'bo-', label="Points")  # 'bo-' for blue points and lines
    ax.set_xlim(0, 6)  # Set x-axis range
    ax.set_ylim(0, 30)  # Set y-axis range
    ax.set_title("Dynamic Point Plotting")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()

    # Plot points one by one
    for i in range(len(x_points)):
        line.set_xdata(x_points[:i+1])  # Update x data
        line.set_ydata(y_points[:i+1])  # Update y data
        plt.draw()  # Redraw the plot
        plt.pause(0.5)  # Pause for 0.5 seconds to simulate delay

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after the loop ends

if __name__ == "__main__":
    main()
