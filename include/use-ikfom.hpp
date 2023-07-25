#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "common_lib.h"
#include "sophus/so3.h"

//This hpp contains, among other things, the definition of the state variable x, the input u, and the function of the correlation matrix in forward propagation.

//24-dimensional state quantities x
struct state_ikfom
{
	Eigen::Vector3d pos = Eigen::Vector3d(0,0,0);
	Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
	Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
	Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d vel = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d bg = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d ba = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d grav = Eigen::Vector3d(0,0,-G_m_s2);
};


// Enter u
struct input_ikfom
{
	Eigen::Vector3d acc = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d gyro = Eigen::Vector3d(0,0,0);
};


// Initialization of the noise covariance Q (corresponding to Q in Eq. (8), used in IMU_Processing.hpp)
Eigen::Matrix<double, 12, 12> process_noise_cov()
{
	Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
	Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

	return Q;
}

//corresponds to f in Eq. (2)
Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in)	
{
// Corresponding order is velocity(3), angular velocity(3), external parameter T(3), external parameter rotation R(3), acceleration(3), angular velocity bias(3), acceleration bias(3), position(3), which is not in the order of the paper's equations.
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
Eigen::Vector3d omega = in.gyro - s.bg; // Angular velocity of the input imu (aka actual measurement) - estimated bias value (corresponds to line 1 of the formula)
Eigen::Vector3d a_inertial = s.rot.matrix() * (in.acc - s.ba); // the acceleration of the input imu, first transferred to the world coordinate system (corresponds to line 3 of the formula)

	for (int i = 0; i < 3; i++)
	{
res(i) = s.vel[i]; //velocity (corresponds to line 2 of the equation)
res(i + 3) = omega[i]; // angular velocity (corresponds to line 1 of the equation)
res(i + 12) = a_inertial[i] + s.grav[i]; //acceleration (corresponds to line 3 of the equation)
	}

	return res;
}

// corresponds to Fx of Eq. (7) Note that the matrix is not multiplied by dt and no unit array is added.
Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity(); //corresponds to Eq. (7), row 2, column 3 I
Eigen::Vector3d acc_ = in.acc - s.ba; //measured acceleration = a_m - bias

cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3::hat(acc_); //corresponds to equation (7), row 3, column 1
cov.block<3, 3>(12, 18) = -s.rot.matrix(); //corresponds to row 3, column 5 of equation (7)

cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); //corresponds to equation (7) row 3 column 6 I
cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); //corresponds to Equation (7), row 1, column 4 (simplified to -I)
	return cov;
}

// corresponds to Fw of Eq. (7) Note that this matrix is not multiplied by dt
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
cov.block<3, 3>(12, 3) = -s.rot.matrix(); //corresponds to equation (7), row 3, column 2 -R
cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity(); //corresponds to Eq. (7), row 1, column 1 -A(w dt) simplifies to -I
cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); //corresponds to Equation (7), row 4, column 3 I
cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); //corresponds to Equation (7), row 5, column 4 I
	return cov;
}

#endif