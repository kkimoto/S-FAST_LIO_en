#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

#include "use-ikfom.hpp"
#include "esekfom.hpp"

/*
This hpp contains mainly:
IMU data preprocessing: IMU initialization, IMU forward propagation, back propagation to compensate for motion distortion
*/

#define MAX_INI_COUNT (10)  //Maximum number of iterations
//Determine the temporal order of the points (note that the timestamp is stored in the curvature)
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
  Eigen::Matrix<double, 12, 12> Q;    //Noise covariance matrix Corresponds to Q in equation (8) of the paper
  void Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_un_);

  V3D cov_acc;             //acceleration covariance
  V3D cov_gyr;             //angular velocity covariance
  V3D cov_acc_scale;       //External Incoming Initial Acceleration Covariance
  V3D cov_gyr_scale;       //External Incoming Initial Angular Velocity Covariance
  V3D cov_bias_gyr;        //Covariance of angular velocity bias
  V3D cov_bias_acc;        //Covariance of acceleration bias
  double first_lidar_time; //Time of the first point cloud of the current frame

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;        //Current frame point cloud is not de-distorted
  sensor_msgs::ImuConstPtr last_imu_;     // Previous frame imu
  vector<Pose6D> IMUpose;                 // Storing imu positions (for back propagation)
  M3D Lidar_R_wrt_IMU;                    // rotational external reference from lidar to IMU
  V3D Lidar_T_wrt_IMU;                    // Lidar to IMU translational outer reference
  V3D mean_acc;                           //Acceleration mean, used to calculate variance
  V3D mean_gyr;                           //Mean value of angular velocity, used to calculate variance
  V3D angvel_last;                        //Angular velocity of the previous frame
  V3D acc_s_last;                         //Previous frame acceleration
  double start_timestamp_;                //Start timestamp
  double last_lidar_end_time_;            //Previous frame end timestamp
  int init_iter_num = 1;                  //Initialize the number of iterations
  bool b_first_frame_ = true;             //Is it the first frame
  bool imu_need_init_ = true;             //Is it necessary to initialize imu
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;                          //Initialize the number of iterations
  Q = process_noise_cov();                    //Call process_noise_cov inside use-ikfom.hpp to initialize the noise covariance
  cov_acc = V3D(0.1, 0.1, 0.1);               //Acceleration covariance initialization
  cov_gyr = V3D(0.1, 0.1, 0.1);               //Angular velocity covariance initialization
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001); //Angular velocity bias covariance initialization
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001); //Acceleration bias covariance initialization
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;                       //Previous frame angular velocity initialization
  Lidar_T_wrt_IMU = Zero3d;                   // Lidar to IMU location external reference initialization
  Lidar_R_wrt_IMU = Eye3d;                    // Initialization of rotational external reference from lidar to IMU
  last_imu_.reset(new sensor_msgs::Imu());    //Previous frame imu initialization
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()   //Reset parameters
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;                   //Is it necessary to initialize imu
  start_timestamp_ = -1;                   //Start timestamp
  init_iter_num = 1;                       //Initialize the number of iterations
  IMUpose.clear();                         // imu bitmap clearing
  last_imu_.reset(new sensor_msgs::Imu()); //Previous frame imu initialization
  cur_pcl_un_.reset(new PointCloudXYZI()); //The current frame point cloud is not de-distorted for initialization.
}

//Passing in external parameters
void ImuProcess::set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
  cov_gyr_scale = gyr;
  cov_acc_scale = acc;
  cov_bias_gyr = gyr_bias;
  cov_bias_acc = acc_bias;
}


//IMU initialization: initialize the state quantity x using the average of the starting IMU frames
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
{
  //MeasureGroup is a struct that represents all the data being processed in the current process, including the IMU queue and the point cloud of a lidar frame, as well as the start and end time of the lidar.
  //Initialize gravity, gyro bias, acc, and gyro covariance Normalize acceleration measurements to unit gravity **/
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) //If for the first frame IMU
  {
    Reset();    //Reset IMU parameters
    N = 1;      //Set the number of iterations to 1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    //Acceleration at the initial moment of the IMU
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       //Angular velocity at the initial moment of IMU
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //The first frame acceleration value is used as the initialized mean value
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              //The first frame angular velocity value is used as the initialization mean value
    first_lidar_time = meas.lidar_beg_time;                   //Use the lidar start time of the current IMU frame as the initial time.
  }

  for (const auto &imu : meas.imu)    //Calculate mean and variance based on all IMU data
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc  += (cur_acc - mean_acc) / N;    //Updates based on current frame and mean difference as a mean
    mean_gyr  += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);

    N ++;
  }
  
  state_ikfom init_state = kf_state.get_x();        //Get the status of x_ at esekfom.hpp
  init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    //Unit direction vector to average the measurements * Gravity acceleration preset value
  
  init_state.bg  = mean_gyr;      //Angular velocity measurement as gyroscopic deviation
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;      //Pass in the lidar and imu external parameters.
  init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
  kf_state.change_x(init_state);      //Pass the initialized state into esekfom.hpp in the x_

  Matrix<double, 24, 24> init_P = MatrixXd::Identity(24,24);      //Obtain the covariance matrix of P_ at esekfom.hpp
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

  // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

//backward propagation
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
  /*** Add the imu from the last end of the previous frame to the imu of the head of the current frame ***/
  auto v_imu = meas.imu;         //Fetch the IMU queue for the current frame
  v_imu.push_front(last_imu_);   //Add the imu at the end of the last frame to the imu at the head of the current frame
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //Time to get the imu at the end of the current frame
  const double &pcl_beg_time = meas.lidar_beg_time;      // Timestamps for the start and end of the point cloud
  const double &pcl_end_time = meas.lidar_end_time;
  
  // Reorder the point cloud based on the timestamp of each point in the point cloud
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  //Here curvature holds the timestamp (in preprocess.cpp)


  state_ikfom imu_state = kf_state.get_x();  // Obtain the posterior state of the last KF estimation as the initial state for this IMU prediction
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
  //The initial state is added to the IMUpose, which contains the time interval, the last frame acceleration, the last frame angular velocity, the last frame velocity, the last frame position, and the last frame rotation matrix.

  /*** Forward propagation ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu; // angvel_avr is the average angular velocity, acc_avr is the average acceleration, acc_imu is the imu acceleration, vel_imu is the imu velocity, pos_imu is the imu position
  M3D R_imu;    //IMU Rotation Matrix Used to eliminate motion distortion.

  double dt = 0;

  input_ikfom in;
  // Iterate over all IMU measurements for this estimation and integrate, discrete median method Forward propagation
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);        //Get the imu data of the current frame
    auto &&tail = *(it_imu + 1);    //Get the imu data for the next frame
    //Determine the time sequence: whether the next frame timestamp is less than the end timestamp of the previous frame or not.
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // median point
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr  = acc_avr * G_m_s2 / mean_acc.norm(); //Adjustment of acceleration by gravity value (in addition to the initialized IMU size * 9.8)

    //If the IMU start moment is earlier than the last radar latest moment (this will happen once because the last IMU from last time was inserted into the beginning of this time)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //Propagation from the end of the last radar moment Calculate the time difference between the end of the IMU and the end of this IMU
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();     //Time interval between two IMU moments
    }
    
    in.acc = acc_avr;     // The median of the two IMU frames is used as inputin for forward propagation
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;         // Configuring the covariance matrix
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state.predict(dt, Q, in);    // IMU forward propagation with a time interval of dt for each propagation

    imu_state = kf_state.get_x();   //Updating the IMU status to the status after integration
    //Update previous frame angular velocity = next frame angular velocity - bias
    angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
    //Update the acceleration in the world coordinate system from the previous frame = R*(acceleration - bias) - g
    acc_s_last  = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();   

    // std::cout << "acc_s_last: " << acc_s_last.transpose() << std::endl;
    // std::cout << "imu_state.ba: " << imu_state.ba.transpose() << std::endl;
    // std::cout << "imu_state.grav: " << imu_state.grav.transpose() << std::endl;
    acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
    // std::cout << "--acc_s_last: " << acc_s_last.transpose() << std::endl<< std::endl;

    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    //The time interval between the latter IMU moment and the start of this radar
    IMUpose.push_back( set_pose6d( offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix() ) );
  }

  // Fill in the last frame of IMU measurements as well
  dt = abs(pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  imu_state = kf_state.get_x();   
  last_imu_ = meas.imu.back();              //Saving the last IMU measurement for the next frame
  last_lidar_end_time_ = pcl_end_time;      //Saves the end time of the last radar measurement of this frame for the next frame

   /*** Elimination of distortion (back propagation) at each LIDAR point ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;

  //Iterate through each IMU frame
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);   //Get the IMU rotation matrix of the previous frame
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);     //Get the IMU speed of the previous frame
    pos_imu<<VEC_FROM_ARRAY(head->pos);     //Get the IMU position of the previous frame
    acc_imu<<VEC_FROM_ARRAY(tail->acc);     //Get the IMU acceleration of the latter frame
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  //Get the IMU angular velocity for the latter frame

    //The point cloud has been sorted by time from smallest to largest before, and IMUpose is similarly pushed in by time from smallest to largest
    //In this case, the loop starts from the end of the IMUpose, that is, from the time maximum, so you only need to judge that the point cloud time needs to be > IMU head moment, that is, you do not need to judge that the point cloud time < IMU tail.
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;    //Interval from point to IMU start moment

      /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix() );   //Rotation at the moment where the point it_pcl is: IMU rotation matrix of the previous frame * exp(angular velocity of the next frame * dt)
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   //Position of the point at the moment (in the radar coordinate system)
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   //World position from point - end of radar world position
      V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}


double T1,T2;
void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &cur_pcl_un_)
{
  // T1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)   
  {
    // The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);  //If the first few frames, you need to initialize the IMU parameters.

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_); 

  // T2 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<T2 - T1<<endl;
}
