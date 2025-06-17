import pathlib
import argparse

from world import generate_random_point, generate_world
from controller import Rob_controller
from robot import Rob_body
from helper import RunningStatsState

from policy_networks import *


parser = argparse.ArgumentParser(description="inputs for robot environment",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--mode", required=True, type=str, default="train", help="do you want to train or do inference")
parser.add_argument("-pre", "--pretrained", required=False, default=False, action="store_true", help="do want to use a pretrained model or train from scratch?")
parser.add_argument("-po", "--policy", required=False, type=str, default="dqn_simple",  help="which policy network do you want to use?")
parser.add_argument("-b", "--batch_size", required=False, type=int, default=256,  help="batch size to train policy with")
parser.add_argument("-s", "--simulations", required=False, type=int, default=256*100,  help="number of simulations to run in training sequence")
parser.add_argument("-f", "--fuel", required=False, type=int, default=300,  help="how many steps the robot can maximally make before running out of fuel")
parser.add_argument("-n", "--n_walls", required=False, type=int, default=3,  help="how many walls do you want to create in the robot world?")
parser.add_argument("-seq", "--sequence_length", required=False, type=int, default=1,  help="sequence length to process. is dependent on the policy chosen")
parser.add_argument("-mem", "--mem_length", required=False, type=int, default=10000,  help="length of memory buffer to train DQN policy")
parser.add_argument("-e", "--epsilon", required=False, type=float, default=1.0,  help="start of the epsilon which determines the exploitation-exploration trade off")
parser.add_argument("-lr", "--learning_rate", required=False, type=float, default=0.00001,  help="parameter to update weights")
parser.add_argument("-std", "--standardized", required=False, default=False, action="store_true",  help="do you want to standardize inputs before feeding to the model")
parser.add_argument("-w", "--world", required=False, type=str, default="simple",  help="which world should the robot navigate?")
args = parser.parse_args()

print("program gets executed with the following arguments:")
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

w, starting_point = generate_world(mode="simple", n_walls=args.n_walls)
b = Rob_body(w, init_pos=starting_point, fuel_tank=args.fuel)
mid = Rob_controller(b, 
                     policy=get_policy(args.policy), 
                     pretrained=args.pretrained, 
                     path_to_weights=str(pathlib.Path(__file__).parent.resolve())+"/policy_stats/"+args.policy + "/" + args.world, 
                     lr =args.learning_rate, 
                     device=device
                     )

def inf():
    mid.do(w, b, action={"go_to": w.goal}, sequence_length=args.sequence_length, standardized=args.standardized)

def train_policy():

    mid.train_dqn(
        batch_size=args.batch_size, 
        simulations=args.simulations, 
        memory_length=args.mem_length, 
        epsilon=args.epsilon, 
        sequence_length=args.sequence_length, 
        n_walls=args.n_walls, 
        standardized=args.standardized,
        world=args.world
        )

    mid.save_policy_stats(path=str(pathlib.Path(__file__).parent.resolve())+"/policy_stats/"+args.policy + "/" + args.world)


if __name__ == "__main__":

    if args.mode == "train":
        train_policy()
    elif args.mode == "inf":
        inf()