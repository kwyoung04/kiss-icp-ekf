#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <sophus/se3.hpp> 

class KalmanFilter {
public:
    Sophus::SE3d state_estimate;

    KalmanFilter() {
        state_estimate = Sophus::SE3d();
        A.setIdentity();
        Q.setIdentity();
        R.setIdentity();
        H.setIdentity();
        P.setIdentity();
    }

    void KalmanUpdate(const Eigen::VectorXd& imu_data, const Eigen::VectorXd& last_pose) {
        // Prediction
        std::cout << "imu_data:\n" << imu_data << std::endl;
        std::cout << "last_pose:\n" << last_pose << std::endl;
        Eigen::VectorXd x = state_estimate.log(); 
        x = A * x + imu_data;
        state_estimate = Sophus::SE3d::exp(x);

        // Kalman gain calculation
        Eigen::Matrix<double, 6, 6> S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 6, 6> K = P * H.transpose() + S.inverse();

        // Update
        x = K * (last_pose - H * x.matrix());
        state_estimate = Sophus::SE3d::exp(x) * state_estimate;
        P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P;
    }
private:
    Eigen::Matrix<double, 6, 6> A;   // State transition matrix
    Eigen::Matrix<double, 6, 6> Q;   // Process noise covariance
    Eigen::Matrix<double, 6, 6> R;   // Measurement covariance
    Eigen::Matrix<double, 6, 6> H;   // Observation matrix
    Eigen::Matrix<double, 6, 6> P;   // Estimate covariance
};