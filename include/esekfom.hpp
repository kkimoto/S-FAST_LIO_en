#ifndef ESEKFOM_EKF_HPP1
#define ESEKFOM_EKF_HPP1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "use-ikfom.hpp"
#include <ikd-Tree/ikd_Tree.h>

//The hpp mainly contains: generalized addition and subtraction, forward propagation principal function, computation of feature point residuals and their Jacobians, ESKF principal function

const double epsi = 0.001; // ESKF iterations are considered convergent if dx<epsi

namespace esekfom
{
	using namespace Eigen;

	PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));		  //Parameters of the plane of the feature point in the map (unit normal vector of the plane, and the distance from the current point to the plane).
	PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); //Effective Feature Points
	PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //Effective eigenpoints correspond to point normal phases
	bool point_selected_surf[100000] = {1};							  //Determine if it is a valid feature point

	struct dyn_share_datastruct
	{
		bool valid;												   //Whether the number of effective feature points meets the requirements
		bool converge;											   //Whether or not convergence has occurred when iterating
		Eigen::Matrix<double, Eigen::Dynamic, 1> h;				   //Residual (z in equation (14))
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; //Jacobi matrix H (H in equation (14))
	};

	class esekf
	{
	public:
		typedef Matrix<double, 24, 24> cov;				// Covariance matrix of 24X24
		typedef Matrix<double, 24, 1> vectorized_state; // Vector of 24X1

		esekf(){};
		~esekf(){};

		state_ikfom get_x()
		{
			return x_;
		}

		cov get_P()
		{
			return P_;
		}

		void change_x(state_ikfom &input_state)
		{
			x_ = input_state;
		}

		void change_P(cov &input_cov)
		{
			P_ = input_cov;
		}

		//Generalized addition Equation (4)
		state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_)
		{
			state_ikfom x_r;
			x_r.pos = x.pos + f_.block<3, 1>(0, 0);

			x_r.rot = x.rot * Sophus::SO3::exp(f_.block<3, 1>(3, 0));
			x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3::exp(f_.block<3, 1>(6, 0));

			x_r.offset_T_L_I = x.offset_T_L_I + f_.block<3, 1>(9, 0);
			x_r.vel = x.vel + f_.block<3, 1>(12, 0);
			x_r.bg = x.bg + f_.block<3, 1>(15, 0);
			x_r.ba = x.ba + f_.block<3, 1>(18, 0);
			x_r.grav = x.grav + f_.block<3, 1>(21, 0);

			return x_r;
		}

		//Forward propagation Equation (4-8)
		void predict(double &dt, Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in)
		{
			Eigen::Matrix<double, 24, 1> f_ = get_f(x_, i_in);	  //Equation (3) for f
			Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, i_in); //df/dx of equation (7)
			Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, i_in); //df/dw of equation (7)

			x_ = boxplus(x_, f_ * dt); //Forward propagation Equation (4)

			f_x_ = Matrix<double, 24, 24>::Identity() + f_x_ * dt; //The entries in the Fx matrix were not multiplied by dt, so I'll add them here.

			P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose(); //The propagation covariance matrix, i.e., equation (8)
		}

		//Calculate the residuals and H-matrix for each feature point
		void h_share_model(dyn_share_datastruct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
						   KD_TREE<PointType> &ikdtree, vector<PointVector> &Nearest_Points, bool extrinsic_est)
		{
			int feats_down_size = feats_down_body->points.size();
			laserCloudOri->clear();
			corr_normvect->clear();

#ifdef MP_EN
			omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif

			for (int i = 0; i < feats_down_size; i++) //Iterate over all feature points
			{
				PointType &point_body = feats_down_body->points[i];
				PointType point_world;

				V3D p_body(point_body.x, point_body.y, point_body.z);
				//The points in the Lidar coordinate system are first transferred to the IMU coordinate system, and then to the world coordinate system based on the forward propagation estimate of the bit position x
				V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
				point_world.x = p_global(0);
				point_world.y = p_global(1);
				point_world.z = p_global(2);
				point_world.intensity = point_body.intensity;

				vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
				auto &points_near = Nearest_Points[i]; // Nearest_Points[i] prints out and finds a vector in order of distance from point_world, from smallest to largest

				double ta = omp_get_wtime();
				if (ekfom_data.converge)
				{
					//Find the nearest neighbor of point_world's planar point
					ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
					//Determine whether a match is valid, similar to the loam series, requires that the number of nearest neighbor map points of a feature point is > threshold, and the distance is < threshold. Only those that satisfy the conditions are set to true.
					point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
																																		: true;
				}
				if (!point_selected_surf[i])
					continue; //If the point does not fulfill the conditions, do not proceed with the following steps.

				Matrix<float, 4, 1> pabcd;		//Plane point information
				point_selected_surf[i] = false; //Setting the point as invalid is used to determine whether the condition is met or not
				//Fit the plane equation ax+by+cz+d=0 and solve for the point-to-plane distance
				if (esti_plane(pabcd, points_near, 0.1f))
				{
					float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); //Distance from the current point to the plane
					float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());												   //If the residual is greater than the empirical threshold, the point is considered valid In short, the closer the lidar point is to the origin, the more demanding the distance from the point to the plane is required

					if (s > 0.9) //If the residual is greater than the threshold, the point is considered valid
					{
						point_selected_surf[i] = true;
						normvec->points[i].x = pabcd(0); //Stores the unit normal vector of the plane and the distance from the current point to the plane.
						normvec->points[i].y = pabcd(1);
						normvec->points[i].z = pabcd(2);
						normvec->points[i].intensity = pd2;
					}
				}
			}

			int effct_feat_num = 0; //Number of effective feature points
			for (int i = 0; i < feats_down_size; i++)
			{
				if (point_selected_surf[i]) //For points that meet the requirements
				{
					laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; //Re-store these points in laserCloudOri
					corr_normvect->points[effct_feat_num] = normvec->points[i];			//Store the normal vectors and distances to the plane corresponding to these points
					effct_feat_num++;
				}
			}

			if (effct_feat_num < 1)
			{
				ekfom_data.valid = false;
				ROS_WARN("No Effective Points! \n");
				return;
			}

			// Computation of Jacobi matrix H and residual vector
			ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
			ekfom_data.h.resize(effct_feat_num);

			for (int i = 0; i < effct_feat_num; i++)
			{
				V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
				M3D point_crossmat;
				point_crossmat << SKEW_SYM_MATRX(point_);
				V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
				M3D point_I_crossmat;
				point_I_crossmat << SKEW_SYM_MATRX(point_I_);

				// Get the normal vector of the corresponding plane
				const PointType &norm_p = corr_normvect->points[i];
				V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

				// Compute the Jacobi matrix H
				V3D C(x_.rot.matrix().transpose() * norm_vec);
				V3D A(point_I_crossmat * C);
				if (extrinsic_est)
				{
					V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
				}
				else
				{
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
				}

				//Residuals: point-plane distances
				ekfom_data.h(i) = -norm_p.intensity;
			}
		}

		//Generalized subtraction
		vectorized_state boxminus(state_ikfom x1, state_ikfom x2)
		{
			vectorized_state x_r = vectorized_state::Zero();

			x_r.block<3, 1>(0, 0) = x1.pos - x2.pos;

			x_r.block<3, 1>(3, 0) = Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
			x_r.block<3, 1>(6, 0) = Sophus::SO3(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();

			x_r.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
			x_r.block<3, 1>(12, 0) = x1.vel - x2.vel;
			x_r.block<3, 1>(15, 0) = x1.bg - x2.bg;
			x_r.block<3, 1>(18, 0) = x1.ba - x2.ba;
			x_r.block<3, 1>(21, 0) = x1.grav - x2.grav;

			return x_r;
		}

		// ESKF
		void update_iterated_dyn_share_modified(double R, PointCloudXYZI::Ptr &feats_down_body,
												KD_TREE<PointType> &ikdtree, vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est)
		{
			normvec->resize(int(feats_down_body->points.size()));

			dyn_share_datastruct dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;
			state_ikfom x_propagated = x_; //Here x_ and P_ are the state quantities and covariance matrix after forward propagation, respectively, since the predict function will be called before this one
			cov P_propagated = P_;

			vectorized_state dx_new = vectorized_state::Zero(); // Vector of 24X1

			for (int i = -1; i < maximum_iter; i++) // maximum_iter is the maximum number of iterations for Kalman filtering
			{
				dyn_share.valid = true;
				// Calculate the Jacobian, which is the derivative of the point surface residual H (h_x in the code)
				h_share_model(dyn_share, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

				if (!dyn_share.valid)
				{
					continue;
				}

				vectorized_state dx;
				dx_new = boxminus(x_, x_propagated); //x^k - x^ in Eq. (18)

				//Since the H-matrix is sparse, only the first 12 columns have non-zero elements, and the last 12 columns are zeros Therefore, here it is computed in the form of a chunked matrix Reduced computation
				auto H = dyn_share.h_x;												// m X 12 matrix
				Eigen::Matrix<double, 24, 24> HTH = Matrix<double, 24, 24>::Zero(); //Matrix H^T * H
				HTH.block<12, 12>(0, 0) = H.transpose() * H;

				auto K_front = (HTH / R + P_.inverse()).inverse();
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
				K = K_front.block<24, 12>(0, 0) * H.transpose() / R; //Kalman gain Here R is considered a constant

				Eigen::Matrix<double, 24, 24> KH = Matrix<double, 24, 24>::Zero(); //Matrix K * H
				KH.block<24, 12>(0, 0) = K * H;
				Matrix<double, 24, 1> dx_ = K * dyn_share.h + (KH - Matrix<double, 24, 24>::Identity()) * dx_new; //Official (18)
				// std::cout << "dx_: " << dx_.transpose() << std::endl;
				x_ = boxplus(x_, dx_); //Official (18)

				dyn_share.converge = true;
				for (int j = 0; j < 24; j++)
				{
					if (std::fabs(dx_[j]) > epsi) //If dx>epsi, it is not convergent.
					{
						dyn_share.converge = false;
						break;
					}
				}

				if (dyn_share.converge)
					t++;

				if (!t && i == maximum_iter - 2) //If the convergence is not achieved after 3 iterations, force true, and the h_share_model function will search for nearest neighbors again.
				{
					dyn_share.converge = true;
				}

				if (t > 1 || i == maximum_iter - 1)
				{
					P_ = (Matrix<double, 24, 24>::Identity() - KH) * P_; //Formulas(19)
					return;
				}
			}
		}

	private:
		state_ikfom x_;
		cov P_ = cov::Identity();
	};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP1
