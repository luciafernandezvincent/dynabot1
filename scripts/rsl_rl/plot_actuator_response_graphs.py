import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    joint_names = [('base_to_front_right_shoulder', 'FRshoulder'),
                   ('front_right_shoulder_to_hand', 'FRarm'),
                   ('front_right_arm_to_hand', 'FRfoot')]
    
    
    
    for sim_name, real_name in joint_names:
        sim_csv = f"actuator_response_results/actuator_response_{sim_name}.csv"
        real_csv = None #VERRR
        
        if real_csv is None:
            output_dir = "actuator_response_results/sim/"
            df_sim = pd.read_csv(sim_csv)
            plt.figure(figsize=(10, 5))
            plt.plot(df_sim["time"], df_sim["joint_pos"], label="joint_pos")

            if "joint_pos_target" in df_sim.columns:
                plt.plot(df_sim["time"], df_sim["joint_pos_target"], label="joint_pos_target")

            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            plt.title(f"Position response - {sim_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sim_name}_position_response.png"), dpi=300)
            plt.close()
                

        else:
            output_dir = "actuator_response_results/sim_real/"
            df_real = pd.read_csv(real_csv)
            df_sim = pd.read_csv(sim_csv)

            #ver como alinear ambos graficos
    
if __name__=='__main__':
    main()