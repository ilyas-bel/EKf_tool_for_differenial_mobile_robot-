#include "ekf.hpp"
#include <cmath> 

ekf::ekf() {
    X.setZero();
    P.setIdentity();
    A.setIdentity();
    B.setZero();
    Q.setIdentity();
    C.setZero();
    R.setIdentity();
    G.setZero();
}

void ekf::predict(double ax, double ay, double wz, double dt) {
    const Eigen::Matrix<double, 7, 1> prev_state = X;

    const double ct = std::cos(prev_state(0));
    const double st = std::sin(prev_state(0));
    const double gx = ax - prev_state(4);
    const double gy = ay - prev_state(5);

    X(0) = prev_state(0) + dt * prev_state(3);          // theta
    X(1) = prev_state(1) + dt * (ct * gx - st * gy);    // vx
    X(2) = prev_state(2) + dt * (st * gx + ct * gy);    // vy
    X(3) = wz - prev_state(6);                          // wz
    X(4) = prev_state(4);                               // bias_ax
    X(5) = prev_state(5);                               // bias_ay
    X(6) = prev_state(6);                               // bias_wz

    // state transition Jacobian
    A.setIdentity();
    A(0, 3) = dt;
    A(1, 0) = -dt * (st * gx + ct * gy);
    A(1, 4) = -dt * ct;   A(1, 5) =  dt * st;
    A(2, 0) =  dt * (ct * gx - st * gy);
    A(2, 4) = -dt * st;   A(2, 5) = -dt * ct;
    A(3, 3) = 0.0;        A(3, 6) = -1.0;

    // noise Jacobian
// noise Jacobian (+noise on every input)
G(1,0) =  dt*ct;  G(1,1) = -dt*st;  G(1,3) = -dt*ct;  G(1,4) =  dt*st;
G(2,0) =  dt*st;  G(2,1) =  dt*ct;  G(2,3) = -dt*st;  G(2,4) = -dt*ct;
G(3,2) =  1.0;                                         G(3,5) = -1.0;
G(4,3) =  1.0;
G(5,4) =  1.0;
G(6,5) =  1.0;

    P = A * P * A.transpose() + G * Q * G.transpose() * dt;
    P = 0.5 * (P + P.transpose());
}

void ekf::update(double v_right, double v_left) {
    const double ct = std::cos(X(0));
    const double st = std::sin(X(0));

    const double vk = X(1) * ct + X(2) * st;

    Eigen::Matrix<double, 2, 1> h;
    h << vk + X(3) * L / 2, vk - X(3) * L / 2;

    const Eigen::Matrix<double, 2, 1> innovation(v_right - h(0), v_left - h(1));

    C(0, 0) = -st * X(1) + ct * X(2);
    C(0, 1) = ct;
    C(0, 2) = st;
    C(0, 3) = L / 2;
    C(1, 0) = -st * X(1) + ct * X(2);
    C(1, 1) = ct;
    C(1, 2) = st;
    C(1, 3) = -L / 2;

    S = C * P * C.transpose() + R;

    if (innovation.dot(S.inverse() * innovation) > 9.2) {
        reject_measurement = true;
        return;   // skip this update, X and P stay unchanged
    }
    reject_measurement = false;

    const Eigen::Matrix<double, 7, 2> K = P * C.transpose() * S.inverse();

    X = X + K * innovation;
    const Eigen::Matrix<double, 7, 7> I_KC = I - K * C;
    P = I_KC * P * I_KC.transpose() + K * R * K.transpose();
    P = 0.5 * (P + P.transpose());
}

void ekf::set_bias(double bias_ax, double bias_ay, double bias_wz) {
    X(4) = bias_ax;
    X(5) = bias_ay;
    X(6) = bias_wz;
}

void ekf::set_initial_state(double theta, double vx, double vy, double wz,
                            double bias_ax, double bias_ay, double bias_wz) {
    X << theta, vx, vy, wz, bias_ax, bias_ay, bias_wz;
}

void ekf::set_R(double r1, double r2) {
    R(0, 0) = r1;
    R(1, 1) = r2;
}

void ekf::set_Q(double q1, double q2, double q3, double q4, double q5, double q6) {
    Q(0, 0) = q1;
    Q(1, 1) = q2;
    Q(2, 2) = q3;
    Q(3, 3) = q4;
    Q(4, 4) = q5;
    Q(5, 5) = q6;
}

void ekf::set_distance_between_wheels(double L_) {
    L = L_;
}