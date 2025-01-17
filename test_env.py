from Simulator.gui.sim import Simulator

env = Simulator()
env.init_window()
env.reset()
while True:
    env.step(0)
    # print(sim.get_robot_data())