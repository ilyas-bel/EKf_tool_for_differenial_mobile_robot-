#pragma once

#include <Eigen/Dense>

class ekf {
public:
    Eigen::Matrix<double, 7, 1> X; // vector state = {theta ,vx ,vy , wz , bias_ax , bias_ay , bias_wz }
    Eigen::Matrix<double, 7, 7> P; // covariance state 
    Eigen::Matrix<double, 7, 7> A; // state transition matrix 
    Eigen::Matrix<double, 7, 3> B; // control input matrix 
    Eigen::Matrix<double, 6, 6> Q; // process covariance matrix
    Eigen::Matrix<double, 2, 7> C; // measurement matrix
    Eigen::Matrix<double, 2, 2> R; // measurement noise covariance matrix 
    Eigen::Matrix<double, 2 ,2> S; // innovation covariance matrix
    Eigen::Matrix<double, 7 , 6>G; // model noise matrix
    bool reject_measurement = false;
    Eigen::Matrix<double,7,7> I = Eigen::Matrix<double,7,7>::Identity();

    double L ;

    ekf();

    void predict(double ax , double ay , double wz, double dt);
    void update(double v_right , double v_left);
    void set_bias(double bias_ax , double bias_ay , double bias_wz);
    void set_initial_state(double theta , double vx , double vy , double wz , double bias_ax , double bias_ay , double bias_wz);
    void set_R(double r1 , double r2);
    void set_Q(double q1 , double q2 , double q3 , double q4 , double q5 , double q6);
    void set_distance_between_wheels(double L);
};