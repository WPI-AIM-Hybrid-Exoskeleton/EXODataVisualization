from GaitAnaylsisToolkit.Session import ViconGaitingTrial
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from GaitCore.Core import Point, PointArray
import csv
import pandas as pd
from sklearn import linear_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #Change to location of file in your system
    curFile = "/home/nathanielgoldfarb/compareMarkerVsExo/EXODataVisualization/pythonProject/Files/11_12_20_nathaniel_walking_00.csv"
    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=curFile)
    markers = trial.vicon.get_markers()
    markers.smart_sort()
    model = trial.vicon.get_model_output()
    leg = model.get_left_leg()

    hip_torque = leg.hip.moment.z
    knee_torque = leg.knee.moment.z

    hip_force_y = leg.hip.force.y
    knee_force_y = leg.knee.force.y

    hip_force_x = leg.hip.force.x
    knee_force_x = leg.knee.force.x

    hip_force_z = leg.hip.force.z
    knee_force_z = leg.knee.force.z

    hip_angle = leg.hip.angle.x
    knee_angle = leg.knee.angle.x

    joint_values_dict = {}
    joint_values_dict['hip_angle'] = hip_angle
    joint_values_dict["knee_angle"] = knee_angle
    joint_values_dict['hip_force_y'] = hip_force_y
    joint_values_dict["knee_force_y"] = knee_force_y

    joint_values_dict['hip_force_x'] = hip_force_x
    joint_values_dict["knee_force_x"] = knee_force_x

    joint_values_dict['hip_force_z'] = hip_force_z
    joint_values_dict["knee_force_z"] = knee_force_z

    joint_values_dict['hip_torque'] = hip_torque
    joint_values_dict["knee_torque"] = knee_torque


    # array of marks
    femur_side = [markers.get_marker("LFemurSide0"), markers.get_marker("LFemurSide1"),
                  markers.get_marker("LFemurSide2"), markers.get_marker("LFemurSide3")]


    femur_center = [markers.get_marker("LFemurFront0"), markers.get_marker("LFemurFront1"),
                    markers.get_marker("LFemurFront2"), markers.get_marker("LFemurFront3"),
                    markers.get_marker("LFemurBack0"), markers.get_marker("LFemurBack1"),
                    markers.get_marker("LFemurBack2"), markers.get_marker("LFemurBack3")]




    tibia_side = [markers.get_marker("LTibiaSide0"), markers.get_marker("LTibiaSide1"),
                  markers.get_marker("LTibiaSide2"), markers.get_marker("LTibiaSide3")]


    tibia_center = [markers.get_marker("LTibiaFront0"), markers.get_marker("LTibiaFront1"),
                    markers.get_marker("LTibiaFront2"), markers.get_marker("LTibiaFront3"),
                    markers.get_marker("LTibiaBack0"), markers.get_marker("LTibiaBack1"),
                    markers.get_marker("LTibiaBack2"), markers.get_marker("LTibiaBack3")]


    trial_range = len(femur_center[0])
    x = np.arange(0, trial_range)

    xav = 0
    yav = 0
    zav = 0
    points = 0
    average_points = []
    average_femur_human = []
    average_femur_side = []

    average_tibia_human = []
    average_tibia_side = []

    for i in range(trial_range):
        human_femur = Point.Point(0, 0, 0)
        exo_femur = Point.Point(0, 0, 0)
        human_tibia = Point.Point(0, 0, 0)
        exo_tibia = Point.Point(0, 0, 0)

        for marker in femur_center:
            human_femur += marker[i]

        for mark in femur_side:
            exo_femur += mark[i]

        for marker in tibia_center:
            human_tibia += marker[i]

        for mark in tibia_side:
            exo_tibia += mark[i]

        human_femur = human_femur / len(femur_center)
        exo_femur = exo_femur / len(femur_side)

        average_femur_human.append(human_femur)
        average_femur_side.append(exo_femur)


        human_tibia = human_tibia / len(tibia_center)
        exo_tibia = exo_tibia / len(tibia_side)

        average_tibia_human.append(human_tibia)
        average_tibia_side.append(exo_tibia)



    average_femur_side = PointArray.PointArray.from_point_array(average_femur_side)
    average_femur_human = PointArray.PointArray.from_point_array(average_femur_human)

    average_tibia_side = PointArray.PointArray.from_point_array(average_tibia_side)
    average_tibia_human = PointArray.PointArray.from_point_array(average_tibia_human)


    x_human = average_femur_human.x
    x_exo = average_femur_side.x
    vx_human = np.diff(x_human)/0.01
    vx_exo = np.diff(x_exo )/0.01

    y_human = average_femur_human.y
    y_exo = average_femur_side.y
    vy_human = np.diff(y_human)/0.01
    vy_exo = np.diff(y_exo )/0.01

    my_dict = {}
    start = 1750
    end = 1965
    my_dict["ex"] = -np.asarray(x_exo[start:end]) - np.asarray(x_human[start:end])
    my_dict["dex"] = -np.asarray(vx_exo[start:end]) - np.asarray(vx_human[start:end])
    my_dict["fx"] = -np.array(leg.hip.force.x[start:end])

    df = pd.DataFrame.from_dict(my_dict)

    X = df[['ex']]
    print(X)
    y = df['fx']
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print("Intercept: ", regr.intercept_)
    print("Coefficients:")
    print(regr.coef_)
    print(regr.rank_)

    plt.plot(vx_human[start:end] , hip_force_x[start:end], '.' )
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
